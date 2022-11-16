import torch.nn as nn
import torchvision.models as models

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
    def __init__(self, embedding_dim, fine_tune=False):
        super(ImageEncoder, self).__init__()

        self.cnn = models.inception_v3(pretrained=True, aux_logits=False)

        # Last 3 layers to identity otherwise this model won't work :(
        self.cnn.avgpool = nn.Identity()
        self.cnn.dropout = nn.Identity()
        self.cnn.fc = nn.Identity()

        for p in self.cnn.parameters():  # freeze all of the network (or not)
            p.requires_grad = False

        # Fine tune cnn (calculate gradients for backprop on last two bottlenecks)
        if fine_tune:
            for c in list(self.cnn.children())[-5:]:
                for p in c.parameters():
                    p.requires_grad = True

        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(-1, 64, 2048)
        out = self.dropout(self.fc(out))

        return out


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
