import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, hidden_dim, activation=nn.SiLU):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), activation(), nn.Linear(hidden_dim, time_dim)
        )

    def forward(self, t):
        return self.time_embed(t)


class TimeMLP(nn.Module):
    def __init__(self, time_dim, hidden_dim, activation=nn.SiLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim * 2),
            activation(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        time_dim: int | None = None,
        activation=nn.ReLU,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── point-wise feed-forward before attention ────────────────────────────
        self.lin1 = nn.Linear(dim, dim)

        # ── multi-head self-attention (batch_first=True) ───────────────────────
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── point-wise feed-forward after attention ───────────────────────────
        self.lin2 = nn.Linear(dim, dim)

        # ── optional time conditioning ─────────────────────────────────────────
        self.time_mlp = None
        if time_dim is not None:
            # re-use your existing TimeMLP utility
            self.time_mlp = TimeMLP(time_dim, dim, activation)

        self.act = activation()

    # --------------------------------------------------------------------- #
    def forward(self, x, time_emb: torch.Tensor | None = None):
        h = self.lin1(x)  # (B, S, D)

        # add conditional bias from the time embedding if provided
        if self.time_mlp is not None and time_emb is not None:
            # broadcast over sequence length
            h = h + self.time_mlp(time_emb).unsqueeze(1)

        h = self.act(h)

        # ── self-attention core ───────────────────────────────────────────────
        # MultiheadAttention expects (B, S, D) when batch_first=True
        attn_out, _ = self.attn(h, h, h)  # (B, S, D)
        h = self.act(attn_out)

        # ── second linear + residual ──────────────────────────────────────────
        h = self.lin2(h)  # (B, S, D)
        return self.act(x + h)  # residual connection


class ScoreNet(nn.Module):
    def __init__(
        self,
        input_dim=1,
        time_dim=96,
        hidden_dim=256,
        num_layers=4,  # This will now be the number of residual blocks
        dtype=torch.float64,
        device=torch.device("cpu"),
        activation=nn.ReLU,
    ):
        super(ScoreNet, self).__init__()

        # Time embedding network
        self.time_embedding = TimeEmbedding(time_dim, hidden_dim, activation)

        # Initial projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Middle layers with residual blocks and time conditioning
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(ResidualBlock(hidden_dim, time_dim, activation))

        # Final layer to get to input_dim
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Skip connections can improve performance
        self.skips = nn.ModuleList()
        if num_layers > 1:
            for _ in range(num_layers // 2):
                self.skips.append(nn.Linear(hidden_dim, hidden_dim))

        self.dtype = dtype
        self.to(device)

        print(f"# Params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x, t):
        # Time conditioning
        time_emb = self.time_embedding(t)

        # Initial projection
        h = self.input_proj(x)
        h = h.unsqueeze(1)

        # Process through residual blocks with time conditioning
        skip_idx = 0
        skip_connections = []

        for i, block in enumerate(self.blocks):
            # Apply residual block with time conditioning
            h = block(h, time_emb)

            # Store and apply skip connections if appropriate
            if i % 2 == 0 and i > 0 and skip_idx < len(self.skips):
                skip_connections.append(h)
            if i % 2 == 1 and len(skip_connections) > 0 and skip_idx < len(self.skips):
                skip_connection = skip_connections.pop()
                h = h + self.skips[skip_idx](skip_connection)
                skip_idx += 1

        # Final projection
        out = self.output_proj(h)
        out.squeeze_(1)  # Remove the sequence dimension

        return out
