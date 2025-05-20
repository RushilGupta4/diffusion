import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ScoreNet(nn.Module):
    """Feed-forward score network for the diffusion model."""

    def __init__(
        self,
        input_dim=1,
        time_embedding=16,
        hidden_dim=128,
        num_layers=3,
        dtype=torch.float64,
    ):
        super(ScoreNet, self).__init__()

        layers = [
            nn.Linear(input_dim + time_embedding, hidden_dim, dtype=dtype),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, dtype=dtype), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, input_dim, dtype=dtype))

        self.time_net = nn.Sequential(
            nn.Linear(1, time_embedding, dtype=dtype),
            nn.ReLU(),
            nn.Linear(time_embedding, time_embedding, dtype=dtype),
            nn.ReLU(),
            nn.Linear(time_embedding, time_embedding, dtype=dtype),
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_net(t)
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)


def get_diffusion_hyperparams(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float64)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def compute_true_score(x, alpha_bar_t, mu, std):
    # For x_t ~ N(sqrt(alpha_bar_t)*mu, (1-alpha_bar_t) + alpha_bar_t*std^2)
    var_t = (1 - alpha_bar_t) + alpha_bar_t * (std**2)
    mean_t = torch.sqrt(alpha_bar_t) * mu
    return -(x - mean_t) / var_t


def main():
    model_path = "model_normal_linear_125.pth"
    T = 1000
    interval = 50  # Interval between time steps
    beta_start = 0.0001
    beta_end = 0.014
    mu = 0.0
    std = np.sqrt(2.0)
    n_points = 1000
    output = "score_comparison.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # Load model
    model = ScoreNet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    # Support both plain state_dict or wrapped in checkpoint
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Diffusion hyperparameters
    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    # Create subplot grid
    num_plots = T // interval + 1  # +1 to include t=0
    n_rows = int(np.ceil(np.sqrt(num_plots)))
    n_cols = int(np.ceil(num_plots / n_rows))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_rows, 4 * n_cols), squeeze=False
    )

    # Prepare inputs
    x = torch.linspace(-10, 10, n_points, device=device, dtype=torch.float64).unsqueeze(
        1
    )

    # Loop through time steps at specified intervals
    plot_idx = 0
    for t_idx in range(0, T + 1, interval):
        # Skip t_idx=T as it's outside the array bounds
        if t_idx == T:
            t_idx = T - 1

        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]

        alpha_bar_t = alpha_bars[t_idx]
        t_norm = torch.full(
            (n_points, 1), t_idx / T, device=device, dtype=torch.float64
        )

        # Compute predicted score
        with torch.no_grad():
            noise_pred = model(x, t_norm)
            score_pred = -noise_pred / torch.sqrt(1 - alpha_bar_t)

        # Compute true score
        score_true = compute_true_score(x, alpha_bar_t, mu, std)

        # Move to numpy
        x_np = x.cpu().numpy().flatten()
        score_pred_np = score_pred.cpu().numpy().flatten()
        score_true_np = score_true.cpu().numpy().flatten()

        ax.plot(x_np, score_true_np, label="True Score")
        ax.plot(x_np, score_pred_np, color="red", label="Predicted Score")

        ax.set_title(f"t={t_idx}/{T}")
        ax.set_xlabel("x")
        ax.set_ylabel("score")

        ax.legend()

        plot_idx += 1

    # Remove empty subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(output)
    print(f"Comparison plot saved to {output}")


if __name__ == "__main__":
    main()
