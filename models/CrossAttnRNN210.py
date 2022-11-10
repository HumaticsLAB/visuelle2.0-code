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
        self.ts_embedder = nn.GRU(1, embedding_dim, batch_first=True)

        # Attention module
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.attentive_aggregator1 = nn.Linear(hidden_dim + 1, hidden_dim)
        self.attentive_aggregator2 = nn.Linear((hidden_dim*2) + 1, hidden_dim)

        # Decoder
        # decoder_input_size = hidden_dim+1 if use_img else hidden_dim
        decoder_input_size = hidden_dim
        self.decoder_gru = nn.GRU(
            input_size=decoder_input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

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
        images,
    ):
        B = X.shape[0]
        img_attn_weights, multimodal_attn_weights = [], [] # Lists to save attention weights
        predictions = [] # List to store predictions of dynamically unrolled outputs.

        # Encode static input data
        img_encoding = self.image_encoder(images)
        
        # Build multimodal input based on specified input modalities
        ts_input = X[:, 0, :].unsqueeze(-1) # Only the first window is selected for successive autoregressive forecasts
        ts_features = self.ts_embedder(ts_input)[1].permute(1,0,2) # Project ts to higher dim space maintaing temporal information

        # Decoder init (first prediction to begin autoregression)
        decoder_hidden = torch.zeros(1, B, self.hidden_dim).to(X.device)
        if self.use_img:
            # Image attention (applied only when necessary)
            attended_img_encoding, img_alpha = self.img_attention(
                img_encoding, decoder_hidden
            )
            img_attn_weights.append(img_alpha)
            attended_img_encoding = attended_img_encoding.sum(1) # Reduce image features via attentive weighted mean
            x = torch.cat([X[:, 0, -1].unsqueeze(-1), attended_img_encoding], dim=-1).unsqueeze(1) # attach last pred to novel image features
            x = self.attentive_aggregator1(x)
        else:
            x = ts_features

        decoder_out, decoder_hidden = self.decoder_gru(x, decoder_hidden)
        pred = self.decoder_fc(decoder_out).squeeze(-1)
         
        # Insert the first prediction and multi-modal attention result.
        predictions.append(pred)

        # Autoregressive rolling forecast
        for t in range(1, self.out_len):
            # Image attention (applied only when necessary)
            if self.use_img:
                attended_img_encoding, img_alpha = self.img_attention(
                    img_encoding, decoder_hidden
                )
                img_attn_weights.append(img_alpha)
                attended_img_encoding = attended_img_encoding.sum(1) # Reduce image features via attentive weighted mean
                x = torch.cat([pred.unsqueeze(1), decoder_out, attended_img_encoding.unsqueeze(1)], dim=-1) # attach last pred to novel image features
                x = self.attentive_aggregator2(x)
            else:
                x = decoder_out                
                
            #### Autoregressive decoding
            decoder_out, decoder_hidden = self.decoder_gru(x, decoder_hidden)
            pred = self.decoder_fc(decoder_out).squeeze(-1)

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
        outputs = torch.stack(predictions).squeeze().T

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
        (X, y, _, _, _, _, _, _), images = train_batch
        forecasted_sales, _, _ = self.forward(
            X,
            y,
            images
        )
        y = y.squeeze()
        loss = F.mse_loss(y, forecasted_sales)
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.use_teacher_forcing = False  # No teacher forcing when evaluating model

    def validation_step(self, test_batch, batch_idx):
        (X, y, _, _, _, _, _, _), images = test_batch
        forecasted_sales, _, _ = self.forward(
            X,
            y,
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

