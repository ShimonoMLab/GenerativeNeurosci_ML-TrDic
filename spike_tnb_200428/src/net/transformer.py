from torch import nn

class TransformerStack(nn.Module):

    def __init__(self, n_input, n_transformer_hidden, n_output, n_heads, n_layers):
        super().__init__()

        self.embedding = nn.Embedding(n_input, n_transformer_hidden)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(n_transformer_hidden, n_heads) for _ in range(n_layers)]
        )
        self.fc = nn.Linear(n_transformer_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        x = embedded.permute(1, 0, 2)  # (seq_len, batch_size, n_transformer_hidden)

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.permute(1, 0, 2)  # (batch_size, seq_len, n_transformer_hidden)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, n_hidden, n_heads):
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=n_hidden, num_heads=n_heads
        )
        self.layer_norm1 = nn.LayerNorm(n_hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_hidden, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, n_hidden),
        )
        self.layer_norm2 = nn.LayerNorm(n_hidden)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.multihead_attention(x, x, x)
        x += residual

        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x += residual

        return x