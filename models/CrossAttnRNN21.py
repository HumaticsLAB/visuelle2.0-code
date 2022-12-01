import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor
from models.modules import AdditiveAttention, ImageEncoder

class CrossAttnRNN(pl.LightningModule):
    def __init__(
        self,
        attention_dim,
        embedding_dim,
        hidden_dim,
        use_img,
        out_len
    ):
        super().__init__()
        self.out_len = out_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_img = use_img

        # Encoder(s)
        self.image_encoder = ImageEncoder(embedding_dim, fine_tune=True)
        self.ts_embedder = nn.GRU(1, embedding_dim, batch_first=True)

        # Attention modules
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)
        )

        self.save_hyperparameters()

    def forward(self, X, y, images):
        bs, num_ts_splits, timesteps = X.shape[0], X.shape[1], X.shape[2]

        # Encode static input data
        img_encoding = self.image_encoder(images)

        # Temporal data
        ts_input = X.reshape((bs*num_ts_splits, timesteps)).unsqueeze(-1) # Collapse values to make 2-1 (single) predictions for each 2 step embedding
        _, ts_embedding = self.ts_embedder(ts_input) # Project ts to higher dim space by preserving temporal order 

        # Image data (optional)
        x = ts_embedding.squeeze()
        if self.use_img:
            img_encoding = img_encoding.repeat_interleave(num_ts_splits, dim=0)
            x = x + img_encoding.sum(1)

        outputs = self.decoder(x)
        outputs = outputs.reshape(bs, num_ts_splits, -1) # Produce in outputs in the form of BS X Num_predictions X 1

        return outputs, None

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )

        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        (X, y, _, _, _, _, _, _), images = train_batch
        forecasted_sales, _ = self.forward(
            X,
            y,
            images
        )
        y = y.squeeze()
        forecasted_sales = forecasted_sales.squeeze()
        loss = F.mse_loss(y, forecasted_sales)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, test_batch, batch_idx):
        (X, y, _, _, _, _, _, _), images = test_batch
        forecasted_sales, _ = self.forward(
            X,
            y,
            images,
        )

        y = y.squeeze()
        forecasted_sales = forecasted_sales.squeeze()
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
        rescaled_item_sales, rescaled_forecasted_sales = (
            item_sales * 53,
            forecasted_sales * 53,
        )  # 53 is the normalization factor (max of the sales of the training set == stfore_sales_norm_scalar.npy)
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

