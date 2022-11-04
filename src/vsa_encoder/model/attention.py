import torch
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self, in_features: int = 1024,
                 latent_dim: int = 1024,
                 n_heads: int = 5):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.embedded_dim = self.latent_dim // n_heads
        self.n_heads = n_heads

        # q, k, v
        self.qkv_proj = nn.Linear(in_features, 3 * n_heads * self.embedded_dim, bias=False)

        # NOT PERSISTENT!
        self.o_proj = nn.Linear(n_heads * self.embedded_dim, latent_dim, bias=False)

        self.scale = 1 / (self.latent_dim * n_heads) ** 0.5

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x -> [batch_size, n_features, latent_dim]
        batch_size, sequence_length, _ = x.size()

        residual = x
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv: torch.Tensor
        qkv = qkv.reshape(batch_size, sequence_length, self.n_heads, 3 * self.latent_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        # x -> [Batch, SeqLen, Head, Dims

        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits * self.scale
        attention = self.softmax(attn_logits)

        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3)
        # x -> [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, sequence_length, self.n_heads * self.latent_dim)

        output = self.o_proj(values)

        return output


class MLP(nn.Module):
    def __init__(self, in_features: int = 1024,
                 hidden_features: int = 1024,
                 out_features: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.mlp(x)


class AttentionBlock(nn.Module):

    def __init__(self, in_features: int = 1024,
                 latent_dim: int = 1024,
                 n_features: int = 5,
                 dropout: float = 0.0,
                 concat_mode: str = 'sum'):
        super().__init__()

        # Two-layer MLP
        self.mlp = MLP(in_features=in_features,
                       hidden_features=latent_dim,
                       out_features=latent_dim)

        # Support layers
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.mlp_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

        # Concatenation mode
        self.concat_mode = concat_mode
        if self.concat_mode == 'sum':
            # Multihead Attention
            self.self_attn = MultiheadAttention(in_features=in_features,
                                                latent_dim=latent_dim,
                                                n_heads=1)
        elif self.concat_mode == 'concat':
            # Multihead Attention
            self.self_attn = MultiheadAttention(in_features=in_features,
                                                latent_dim=latent_dim,
                                                n_heads=n_features)
        else:
            raise ValueError(f"Wrong concat_mode: {concat_mode}")

    def forward(self, x):
        # Attention part
        attn_out = self.self_attn(x)
        attn_out = torch.squeeze(attn_out, -2)
        x = x + self.dropout(attn_out)
        x = self.attn_norm(x)

        # MLP part
        linear_out = self.mlp(x)
        x = x + self.dropout(linear_out)
        x = self.mlp_norm(x)

        return x


if __name__ == '__main__':
    batch_size = 4
    n_features = 5
    latent_dim = 1024

    attention = AttentionBlock(n_features=n_features,
                               in_features=latent_dim,
                               latent_dim=latent_dim)

    x = torch.randn(batch_size, n_features, latent_dim)

    out = attention(x)

    print("Done")

# TODO: первый вариант - 1 Head, длиной 1024 и сложить признаки
# TODO: второй вариант - 5 Head, при этом конкатенация векторов
