
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()

        assert embed_dim % heads == 0

        self.heads = heads
        self.head_dim = embed_dim // heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch, seq_len, embed_dim = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch, seq_len, 3, self.heads, self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask: prevent looking at future tokens
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = attention @ v

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch, seq_len, embed_dim)

        return self.projection(output)


class FeedForward(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(embed_dim, heads, dropout)

        self.feed_forward = FeedForward(embed_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.attention(self.norm1(x))

        x = x + self.feed_forward(self.norm2(x))

        return x


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=32,
        embed_dim=128,
        heads=4,
        layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            block_size, embed_dim
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, heads, dropout)
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        batch, seq_len = x.shape

        positions = torch.arange(
            seq_len, device=x.device
        )

        token_embeddings = self.token_embedding(x)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.final_norm(x)

        logits = self.output_layer(x)

        return logits

    @torch.no_grad()
    def generate(self, x, max_new_tokens=30, temperature=1.0):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the latest context
            x_context = x[:, -self.block_size:]

            logits = self(x_context)

            # Get predictions for the last token
            logits = logits[:, -1, :]

            logits = logits / max(temperature, 0.01)

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            x = torch.cat((x, next_token), dim=1)

        return x
