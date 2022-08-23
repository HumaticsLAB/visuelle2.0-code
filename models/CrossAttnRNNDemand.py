import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from fairseq.optim.adafactor import Adafactor


class TSEmbedder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TSEmbedder, self).__init__()
        self.ts_embedder = nn.GRU(
            input_size=input_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.ts_embedder(x)[0])

        return x


class AttributeEncoder(nn.Module):
    def __init__(self, num_cat, num_col, num_fab, num_store, embedding_dim):
        super(AttributeEncoder, self).__init__()
        self.cat_embedder = nn.Embedding(num_cat, embedding_dim)
        self.col_embedder = nn.Embedding(num_col, embedding_dim)
        self.fab_embedder = nn.Embedding(num_fab, embedding_dim)
        self.store_embedder = nn.Embedding(num_store, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cat, col, fab, store):
        cat_emb = self.dropout(self.cat_embedder(cat))
        col_emb = self.dropout(self.col_embedder(col))
        fab_emb = self.dropout(self.fab_embedder(fab))
        store_emb = self.dropout(self.store_embedder(store))
        attribute_embeddings = cat_emb + col_emb + fab_emb + store_emb

        return attribute_embeddings


class TemporalFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        d = temporal_features[:, 0].unsqueeze(1)
        w = temporal_features[:, 1].unsqueeze(1)
        m = temporal_features[:, 2].unsqueeze(1)
        y = temporal_features[:, 3].unsqueeze(1)
        d_emb = self.dropout(self.day_embedding(d))
        w_emb = self.dropout(self.day_embedding(w))
        m_emb = self.dropout(self.day_embedding(m))
        y_emb = self.dropout(self.day_embedding(y))
        temporal_embeddings = d_emb + w_emb + m_emb + y_emb

        return temporal_embeddings


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=300):
        super(ImageEncoder, self).__init__()

        ft_ex_modules = list(models.resnet101(pretrained=True).children())[:-2]
        self.cnn = nn.Sequential(*ft_ex_modules)
        for p in self.cnn.parameters():  # freeze all of the network
            p.requires_grad = False

        # Fine tune cnn (calculate gradients for backprop on last two bottlenecks)
        for c in list(self.cnn.children())[6:]:
            for p in c.parameters():
                p.requires_grad = True

        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64, 2048)
        x = self.dropout(self.fc(x))

        return x


class AdditiveAttention(nn.Module):  # Bahdanau encoder-decoder attention (Additive attention)
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attn_linear = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # h_j and s_i refer to the variable names from the original formula of Bahdanau et al.

        h_j = self.encoder_linear(encoder_out)  # (batch_size, len, attention_dim)
        s_i = self.decoder_linear(decoder_hidden).squeeze(
            0
        )  # (batch_size, attention_dim)
        energy = self.attn_linear(self.tanh(h_j + s_i.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, len)
        alpha = self.softmax(energy)  # (batch_size, len)
        attention_weighted_encoding = (
            alpha.unsqueeze(2) * h_j
        )  # (batch_size, len, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x_input, decoder_hidden):

        gru_out, self.hidden = self.gru(x_input, decoder_hidden)
        output = self.linear(gru_out)

        return output, self.hidden


class GatingFCN(nn.Module):
    def __init__(self, input_dim):
        super(GatingFCN, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim, bias=False)
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        res = x
        x = (
            self.gate_fn(self.fc(x)) * x
        )  # gate feature space based on sigmoid activation
        out = res + x  # residual learning

        return out


class CrossAttnRNN(pl.LightningModule):
    def __init__(
        self,
        attention_dim,
        embedding_dim,
        num_trends,
        hidden_dim,
        cat_dict,
        col_dict,
        fab_dict,
        store_num,
        use_img,
        use_att,
        use_date,
        use_trends,
        out_len=12,
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
        self.trend_encoder = TSEmbedder(num_trends, embedding_dim)
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
        self.ts_self_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=0.1)
        self.ts_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)   
        self.trend_linear = nn.Linear(52*attention_dim, embedding_dim)
        self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_embedder = nn.Linear(embedding_dim, embedding_dim)

        # Fully connected network with a gating mechanism for the final output
        self.gate = GatingFCN(attention_dim)

        # Decoder
        self.decoder = nn.GRU(
            input_size=embedding_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_fc = nn.Linear(hidden_dim, 1)

        self.save_hyperparameters()

    def forward(
        self,
        ts,
        categories,
        colors,
        fabrics,
        stores,
        temporal_features,
        gtrends,
        images,
    ):
        bs = ts.shape[0]
        # self.use_teacher_forcing = False
        # print(f"forward call teacher forcing: {self.use_teacher_forcing}")

        # Encode input data
        gtrend_encoding = self.trend_encoder(gtrends.permute(0,2,1))
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.temp_encoder(temporal_features)
        attribute_encoding = self.attribute_encoder(categories, colors, fabrics, stores)

        if self.use_trends:
            # Self-attention over the temporal features (this filters initial noise from the time series features)
            gtrend_encoding, trend_self_attn_weights = self.ts_self_attention(
                gtrend_encoding.permute(1, 0, 2),
                gtrend_encoding.permute(1, 0, 2),
                gtrend_encoding.permute(1, 0, 2),
            )

        # Predictions vector (will contain all forecasts)
        outputs = torch.zeros(bs, self.out_len, 1).to(self.device)

        # Save attention weights
        img_attn_weights, trend_attn_weights, multimodal_attn_weights = [], [], []

        # Init initial decoder status and outputs
        decoder_hidden = torch.zeros(1, bs, self.hidden_dim).to(self.device)
        decoder_output = torch.zeros(bs, 1, 1).to(self.device).to(self.device)
        for t in range(self.out_len):
            # Image attention
            if self.use_img:
                attended_img_encoding, img_alpha = self.img_attention(
                    img_encoding, decoder_hidden
                )
                # Reduce image features into one via summing
                attended_img_encoding = attended_img_encoding.sum(1)
                img_attn_weights.append(img_alpha)


            # Temporal (Exogenous Gtrends) Attention
            if self.use_trends:
                attended_trend_encoding, trend_alpha = self.ts_attention(
                    gtrend_encoding.permute(1,0,2), decoder_hidden
                )
                attended_trend_encoding = self.trend_linear(attended_trend_encoding.view(bs, -1))
                trend_attn_weights.append(trend_alpha)


            # Build multimodal input based on specified input modalities
            mm_in = dummy_encoding.unsqueeze(0)
            if self.use_img:
                mm_in = torch.cat([mm_in, attended_img_encoding.unsqueeze(0)])
            if self.use_att:
                mm_in = torch.cat([mm_in, attribute_encoding.unsqueeze(0)])
            if self.use_trends:
                mm_in = torch.cat([mm_in, attended_trend_encoding.unsqueeze(0)])
            mm_in = mm_in.permute(1, 0, 2)
            
            # Multimodal attention
            attended_multimodal_encoding, multimodal_alpha = self.multimodal_attention(
                mm_in, decoder_hidden
            )

            # Save alphas
            multimodal_attn_weights.append(multimodal_alpha)

            # Gate attention output (WIP)
            # final_attention_encoding = self.gate(attended_multimodal_encoding)

            # Reduce (residual) attention weighted multimodal input via summation and then embed
            final_embedding = mm_in + attended_multimodal_encoding  # residual learning
            final_encoder_output = self.multimodal_embedder(
                final_embedding.sum(1) # reduce sum
            )  # BS X 1 X D

            # Concatenate last predicition to the encoder output -> autoregression
            x_input = torch.cat(
                [final_encoder_output.unsqueeze(1), decoder_output], dim=2
            )

            # GRU decoder
            decoder_out, decoder_hidden = self.decoder(x_input, decoder_hidden)
            decoder_output = self.decoder_fc(decoder_out)
            outputs[:, t, :] = decoder_output[: , 0 , :]

            # Control teacher forcing
            teach_forcing = (
                True if torch.rand(1) < self.teacher_forcing_ratio else False
            )
            if self.use_teacher_forcing and teach_forcing and ts is not None:
                decoder_output = ts[:, t].unsqueeze(-1).unsqueeze(-1)

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
            (ts, categories, colors, fabrics, stores, temporal_features, gtrends),
            images,
        ) = train_batch
        forecasted_sales, _, _ = self.forward(
            ts,
            categories,
            colors,
            fabrics,
            stores,
            temporal_features,
            gtrends,
            images,
        )
        loss = F.mse_loss(ts, forecasted_sales.squeeze())
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.use_teacher_forcing = False  # No teacher forcing when evaluating model

    def validation_step(self, test_batch, batch_idx):
        (
            (ts, categories, colors, fabrics, stores, temporal_features, gtrends),
            images,
        ) = test_batch
        forecasted_sales, _, _ = self.forward(
            ts,
            categories,
            colors,
            fabrics,
            stores,
            temporal_features,
            gtrends,
            images,
        )
        return ts, forecasted_sales

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
        )  # 53 is the normalization factor (max of the sales of the training set)
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

