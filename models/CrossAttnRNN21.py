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
        use_img,
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

        # Encoder(s)
        self.image_encoder = ImageEncoder(embedding_dim)
        self.ts_embedder = nn.GRU(1, embedding_dim, dropout=0.2, batch_first=True)

        # Attention modules
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
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
        bs, num_ts_splits, timesteps = X.shape[0], X.shape[1], X.shape[2]

        # Encode static input data
        img_encoding = self.image_encoder(images)

        # Temporal data
        ts_input = X.reshape((bs*num_ts_splits, timesteps)).unsqueeze(-1) # Collapse values to make 2-1 (single) predictions for each 2 step embedding
        _, ts_embedding = self.ts_embedder(ts_input) # Project ts to higher dim space by preserving temporal order 

        mm_in = ts_embedding.squeeze() 
        if self.use_img:
            img_encoding = img_encoding.repeat_interleave(num_ts_splits, dim=0)

            # Image attention over the temporal embedding
            attended_img_encoding, _ = self.img_attention(
                img_encoding, ts_embedding
            )
            attended_img_encoding = attended_img_encoding.sum(1) # Reduce image features by summing over the attention weighted encoding
            attended_img_encoding = F.dropout(attended_img_encoding, p=0.2, training=self.training) 
            mm_in = torch.cat([mm_in, attended_img_encoding], dim=1)

        outputs = self.decoder(mm_in)
        outputs = outputs.reshape(bs, num_ts_splits, -1) # Produce in outputs in the form of BS X Num_predictions X 1

        return outputs, [], []

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

