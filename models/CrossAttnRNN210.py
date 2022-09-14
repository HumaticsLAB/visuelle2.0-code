import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor
from models.modules import AdditiveAttention, ImageEncoder, AttributeEncoder, TemporalFeatureEncoder

class CrossAttnRNN(pl.LightningModule):
    def __init__(
        self,
        attention_dim,
        embedding_dim,
        hidden_dim,
        cat_dict,
        col_dict,
        fab_dict,
        store_num,
        use_img,
        use_att,
        use_date,
        use_trends,
        task_mode,
        out_len,
        use_teacher_forcing=False,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_teacher_forcing = use_teacher_forcing
        self.out_len = out_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_img = use_img
        self.use_att = use_att
        self.use_date = use_date
        self.use_trends = use_trends

        # Encoder(s)
        self.temp_encoder = TemporalFeatureEncoder(embedding_dim)
        self.image_encoder = ImageEncoder()
        self.attribute_encoder = AttributeEncoder(
            len(cat_dict) + 1,
            len(col_dict) + 1,
            len(fab_dict) + 1,
            store_num + 1,
            embedding_dim,
        )

        # Attention modules
        self.ts_lin = nn.Linear(1, embedding_dim)
        self.ts_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        if self.use_trends:
            self.ts_embedder = nn.Linear(embedding_dim * (156+(12-out_len)), embedding_dim)
        else:
            self.ts_embedder = nn.Linear(embedding_dim * (12-out_len), embedding_dim)
        
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_embedder = nn.Linear(embedding_dim, embedding_dim)

        # Decoder
        self.gru = nn.GRU(
            # input_size=embedding_dim,
            input_size=embedding_dim+1,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=0.2,
            batch_first=True
        )

        # self.decoder_fc = nn.Linear(hidden_dim, 1)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)
        )

        self.save_hyperparameters()

    def forward(
        self,
        X,
        y,
        categories,
        colors,
        fabrics,
        stores,
        temporal_features,
        gtrends,
        images,
    ):
        bs = X.shape[0]
        
        # Encode static input data
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.temp_encoder(temporal_features)
        attribute_encoding = self.attribute_encoder(categories, colors, fabrics, stores)
        
        # Encode temporal variables (past sales, exogenous variables) into a single feature
        flatten_exo = gtrends.view(
            bs, -1, 1
        )  # Flatten all temporal features into one series
        flatten_exo = torch.repeat_interleave(
            flatten_exo, X.shape[1], dim=-1
        )  # Repeat the above series N times

        # Decide to use exogenous series or not
        if self.use_trends:
            ts_input = torch.cat(
                [flatten_exo.permute(0,2,1), X], dim=2
            )  # Concatenate all temporal features such that sales are the most recent
        else:
            # ts_input = X.permute(0, 2, 1)
            ts_input = X
        
        # Build multimodal input based on specified input modalities
        ts_input = ts_input[:, 0, :].unsqueeze(-1) # Only the first window is selected for successive autoregressive forecasts
        ts_encoding = self.ts_lin(ts_input) # Project ts to higher dim space
        mm_in = ts_encoding.sum(1).unsqueeze(0)
        if self.use_img:
            mm_in = torch.cat([mm_in, img_encoding.sum(1).unsqueeze(0)])
        if self.use_att:
            mm_in = torch.cat([mm_in, attribute_encoding.unsqueeze(0)])
        if self.use_date:
            mm_in = torch.cat([mm_in, dummy_encoding.unsqueeze(0)])

        mm_in = mm_in.sum(0) # Reduce sum (without attention initially)


        # Store predictions of dynamically unrolled outputs.
        predictions = []
        mm_embeddings = []

        # Decoder warm-up/init (first prediction of autoregression)
        warmup_input = mm_in.unsqueeze(1)
        start_token = torch.zeros(bs, 1, 1).to(self.device)-1
        warmup_input = torch.cat([warmup_input, start_token], dim=-1)
        _, decoder_hidden = self.gru(warmup_input)

        # Insert the first prediction and multi-modal attention result.
        pred = self.decoder_fc(decoder_hidden).squeeze(0)
        predictions.append(pred)

        # Autoregressive rolling forecast
        for t in range(1, self.out_len):
            ### Stepwise additive attention
            img_attn_weights, multimodal_attn_weights = [], [] # Lists to save attention weights

            # Ts Attention (this is always done)
            # TODO: Should this be based on the fully predicted sequence?
            attended_ts_encoding, _ = self.ts_attention(
                ts_encoding, decoder_hidden
            )
            attended_ts_encoding = F.dropout(attended_ts_encoding, p=0.1, training=self.training) 
            attended_ts_encoding = attended_ts_encoding.sum(1) # Reduce TS features via summing
            

            # Image attention (applied only when necessary)
            updated_mm = attended_ts_encoding.unsqueeze(0)
            if self.use_img:
                attended_img_encoding, img_alpha = self.img_attention(
                    img_encoding, decoder_hidden
                )
                img_attn_weights.append(img_alpha)
                attended_img_encoding = F.dropout(attended_img_encoding, p=0.1, training=self.training) 
                attended_img_encoding = attended_img_encoding.sum(1) # Reduce image features via summing
                updated_mm = torch.cat([updated_mm, attended_img_encoding.unsqueeze(0)])

            # Update multimodal with new hidden state
            attended_multimodal_encoding, multimodal_alpha = self.multimodal_attention(
                updated_mm.permute(1,0,2), decoder_hidden  # Change mm embedding to BS x len x D for attention layer
            )
            attended_multimodal_encoding = F.dropout(attended_multimodal_encoding, p=0.5, training=self.training) 
            attended_multimodal_encoding = attended_multimodal_encoding.sum(1) # Reduce over multimodal features via summing
            multimodal_attn_weights.append(multimodal_alpha)

            # Reduce (residual) attention weighted multimodal input via summation and then embed
            final_mm_embedding = mm_in + attended_multimodal_encoding  # residual learning
            mm_embeddings.append(final_mm_embedding)

            # Use the last prediction and multimodal embedding as input for the current step.
            pred_seq = torch.stack(predictions) # t x BS x D
            mm_emb_seq = torch.stack(mm_embeddings) # t x BS x D
            x = torch.cat([mm_emb_seq, pred_seq], dim=-1).permute(1,0,2) # BS x t D

            # Pass through rnn.
            _, decoder_hidden = self.gru(x, decoder_hidden)

            # Make new prediction
            pred = self.decoder_fc(decoder_hidden).squeeze(0)

            # Control teacher forcing
            if self.use_teacher_forcing:
                teach_forcing_prob = True if torch.rand(1) < self.teacher_forcing_ratio else False
                if teach_forcing_prob and y is not None:
                    predictions.append(y[:, t, :])
                else:
                    predictions.append(pred)
            else:
                predictions.append(pred)

         # Convert the RNN output to a single prediction.
        outputs = torch.stack(predictions).permute(1,0,2)

        return outputs, img_attn_weights, multimodal_attn_weights

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )

        return [optimizer]

    def on_train_epoch_start(self):
        self.use_teacher_forcing = True  # Allow for teacher forcing when training model

    def training_step(self, train_batch, batch_idx):
        (
            (X, y, categories, colors, fabrics, stores, temporal_features, gtrends),
            images,
        ) = train_batch
        forecasted_sales, _, _ = self.forward(
            X,
            y,
            categories,
            colors,
            fabrics,
            stores,
            temporal_features,
            gtrends,
            images,
        )
        loss = F.mse_loss(y, forecasted_sales)
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.use_teacher_forcing = False  # No teacher forcing when evaluating model

    def validation_step(self, test_batch, batch_idx):
        (
            (X, y, categories, colors, fabrics, stores, temporal_features, gtrends),
            images,
        ) = test_batch
        forecasted_sales, _, _ = self.forward(
            X,
            y,
            categories,
            colors,
            fabrics,
            stores,
            temporal_features,
            gtrends,
            images,
        )
        return y, forecasted_sales

    def validation_epoch_end(self, val_step_outputs):

        item_sales, forecasted_sales = (
            [x[0] for x in val_step_outputs],
            [x[1] for x in val_step_outputs],
        )
        item_sales, forecasted_sales = (
            torch.vstack(item_sales),
            torch.vstack(forecasted_sales),
        )
        item_sales, forecasted_sales = item_sales.squeeze(), forecasted_sales.squeeze()
        rescaled_item_sales, rescaled_forecasted_sales = (
            item_sales * 53,
            forecasted_sales * 53,
        )  # 53 is the normalization factor (max of the sales of the training set in stfore_sales_norm_scalar.npy)
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        wape = 100 * torch.sum(torch.abs(rescaled_item_sales - rescaled_forecasted_sales)) / torch.sum(rescaled_item_sales)

        self.log("val_mae", mae)
        self.log("val_wWAPE", wape)
        self.log("val_loss", loss)

        print(
            "Validation MAE:",
            mae.detach().cpu().numpy(),
            "Validation WAPE:",
            wape.detach().cpu().numpy(),
            "LR:",
            self.optimizers().param_groups[0]["lr"],
        )

