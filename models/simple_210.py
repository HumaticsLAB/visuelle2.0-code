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
        self.image_encoder = ImageEncoder(embedding_dim)
        self.warmup_rnn = nn.GRU(embedding_dim+1, embedding_dim, dropout=0.1, batch_first=True)
        
        # Attention modules
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_embedder = nn.Linear(embedding_dim, embedding_dim)

        # Decoder
        self.rnn_cell = nn.GRUCell(embedding_dim+1, hidden_dim)
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
        img_attn_weights, multimodal_attn_weights = [], [] # Lists to save attention weights

        # Encode static input data
        img_encoding = self.image_encoder(images)

        # Store predictions of dynamically unrolled outputs.
        predictions = []
        mm_embeddings = []

        # Decoder warm-up/init (first prediction of autoregression)
        start_tokens = X[:, 0, :].unsqueeze(-1) # Only the first window is selected for successive autoregressive forecasts
        img_encoding = img_encoding.mean(1).unsqueeze(1).repeat_interleave(start_tokens.shape[1], dim=1)
        warmup_in = torch.cat([img_encoding, start_tokens], dim=-1)
        warmup_out, decoder_hidden = self.warmup_rnn(warmup_in)
        decoder_hidden = decoder_hidden.squeeze()
        ts_embedding = warmup_out.mean(1)

        # Insert the first prediction and multi-modal attention result.
        pred = self.decoder_fc(decoder_hidden).squeeze(0)
        predictions.append(pred)

        # Autoregressive rolling forecasts
        for t in range(1, self.out_len):
            # Image attention
            if self.use_img:
                attended_img_encoding, img_alpha = self.img_attention(
                    img_encoding, decoder_hidden
                )
                # Reduce image features into one via summing
                img_attn_weights.append(img_alpha)
                attended_img_encoding = F.dropout(attended_img_encoding, p=0.1, training=self.training) 
                attended_img_encoding = attended_img_encoding.sum(1)

            # Build multimodal input based on specified input modalities
            mm_in = ts_embedding.unsqueeze(0)
            if self.use_img:
                mm_in = torch.cat([mm_in, attended_img_encoding.unsqueeze(0)])
            mm_in = mm_in.permute(1, 0, 2)
            
            # Multimodal attention
            attended_multimodal_encoding, multimodal_alpha = self.multimodal_attention(
                mm_in, decoder_hidden  # Change mm embedding to BS x len x D for attention layer
            )
            multimodal_attn_weights.append(multimodal_alpha)
            attended_multimodal_encoding = F.dropout(attended_multimodal_encoding, p=0.1, training=self.training)

            final_mm_embedding = mm_in + attended_multimodal_encoding  # residual learning
            final_mm_embedding = self.multimodal_embedder(final_mm_embedding.sum(1))  # BS X D

            # Update hidden state for sequential predictions
            decoder_hidden = decoder_hidden + final_mm_embedding # Condition hidden state

            # Use the last prediction as input for the current step.
            step_in = torch.cat([final_mm_embedding, pred], dim=-1)
            decoder_hidden = self.rnn_cell(step_in, decoder_hidden)

            # Make new prediction
            pred = self.decoder_fc(decoder_hidden).squeeze(0)

            # Control teacher forcing
            if self.use_teacher_forcing:
                teach_forcing_prob = True if torch.rand(1) < self.teacher_forcing_ratio else False
                if teach_forcing_prob and y is not None:
                    pred = y[:, t, :]
            
            predictions.append(pred)

        # Convert the RNN outputs to a prediction tensor.
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

