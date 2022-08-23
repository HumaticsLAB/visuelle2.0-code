import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
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
        self.ts_attn_lin = nn.Linear(1, embedding_dim)
        self.ts_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)

        if self.use_trends:
            self.ts_embedder = nn.Linear(embedding_dim * (156+(12-out_len)), embedding_dim)
        else:
            self.ts_embedder = nn.Linear(embedding_dim * (12-out_len), embedding_dim)
        
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_embedder = nn.Linear(embedding_dim, embedding_dim)

        # Decoder
        self.decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(hidden_dim, 1)

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
        bs, num_ts_splits, timesteps = X.shape[0], X.shape[1], X.shape[2]

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

        # Predictions vector (will contain all forecasts)
        outputs = torch.zeros(bs, self.out_len, 1).to(self.device)

        # Save attention weights
        img_attn_weights, multimodal_attn_weights = [], []

        # Temporal data
        # Decide to use exogenous series or not
        if self.use_trends:
            ts_input = torch.cat(
                [flatten_exo.permute(0,2,1), X], dim=2
            )  # Concatenate all temporal features such that sales are the most recent
            timesteps += 156 # 3*52 gtrends
        else:
            # ts_input = X.permute(0, 2, 1)
            ts_input = X


        # 2-1 Single predictions for each 2 step embedding
        # Init initial decoder status
        decoder_hidden = torch.zeros(1, bs*num_ts_splits, self.hidden_dim).to(self.device)

        # TODO: FIX OLD PROBLEMS AND FOUND SOLUTION FOR NEW ONES: COLLAPSE BATCHING
        ts_input = ts_input.reshape((bs*num_ts_splits, timesteps)).unsqueeze(-1)
        ts_input = self.ts_attn_lin(ts_input) # Project ts to higher dim space

        attended_ts_encoding, ts_alpha = self.ts_attention(
            ts_input, decoder_hidden
        )

        # Because here we have much more data samples, we have to repeat the multimodal embeddings to arrive at the same point

        # Build multimodal input based on specified input modalities
        # TODO: How to correctly mix the embeddings for 2-1? 
        ts_encoding = self.ts_embedder(attended_ts_encoding.view(bs*num_ts_splits, -1))
        mm_in = ts_encoding.unsqueeze(0)
        
        if self.use_img:
            img_encoding = img_encoding.repeat_interleave(num_ts_splits, dim=0)
            
            # Image attention
            attended_img_encoding, img_alpha = self.img_attention(
                img_encoding, decoder_hidden
            )
            img_attn_weights.append(img_alpha)

            # Reduce image features via summing
            attended_img_encoding = attended_img_encoding.sum(1)

            mm_in = torch.cat([mm_in, attended_img_encoding.unsqueeze(0)])
        if self.use_att:
            attribute_encoding = attribute_encoding.repeat_interleave(num_ts_splits, dim=0)
            mm_in = torch.cat([mm_in, attribute_encoding.unsqueeze(0)])
        if self.use_date:
            dummy_encoding = dummy_encoding.repeat_interleave(num_ts_splits, dim=0)
            mm_in = torch.cat([mm_in, dummy_encoding.unsqueeze(0)])
        mm_in = mm_in.permute(1, 0, 2)
        
        # Multimodal attention
        attended_multimodal_encoding, multimodal_alpha = self.multimodal_attention(
            mm_in, decoder_hidden
        )
        multimodal_attn_weights.append(multimodal_alpha)

        # Reduce (residual) attention weighted multimodal input via summation and then embed
        final_embedding = mm_in + attended_multimodal_encoding  # residual learning
        final_encoder_output = self.multimodal_embedder(
            final_embedding.sum(1) # reduce sum
        )  # BS X 1 X D

        # 21 input
        # stepwise_in = torch.cat([
        #     torch.repeat_interleave(final_encoder_output.unsqueeze(1), self.out_len, dim=1),
        #     X], dim=-1)
        decoder_out, decoder_hidden = self.decoder(final_encoder_output.unsqueeze(1), decoder_hidden)
        outputs = self.decoder_fc(decoder_out)
        outputs = outputs.reshape(bs, num_ts_splits, -1)

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

