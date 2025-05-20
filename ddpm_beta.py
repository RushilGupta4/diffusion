import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
from typing import List, Tuple

THRESHOLD = 10


class ScoreNet(nn.Module):
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
        self.dtype = dtype

        print(f"# Params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x, t):
        t = self.time_net(t)
        x = torch.cat([x, t], dim=1)
        return self.net(x)


def get_diffusion_hyperparams(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float64)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def create_beta_distribution(num_points=1000, alpha_param=2.0, beta_param=5.0):
    """Generates a beta distribution dataset with given shape parameters."""
    data = np.random.beta(alpha_param, beta_param, num_points)
    return torch.tensor(data, dtype=torch.float64).unsqueeze(1)


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    betas,
    alphas,
    alpha_bars,
    T,
    device,
    alpha_param,
    beta_param,
    theta_list=[0],
    ckpt="model_{step}.pth",
    start_epoch=0,
    num_epochs=10,
    num_samples=50000,
):
    """Training loop for the diffusion model with beta data."""

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_loss = 0.0
        with tqdm(total=num_batches, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for (x0,) in dataloader:
                x0 = x0.to(device)
                B = x0.shape[0]

                t = torch.randint(0, T, (B,), device=device)
                t_norm = t.float() / T
                t_norm = t_norm.unsqueeze(1).to(torch.float64)

                noise = torch.randn_like(x0)
                x_t = (
                    torch.sqrt(alpha_bars[t]).unsqueeze(1) * x0
                    + torch.sqrt(1 - alpha_bars[t]).unsqueeze(1) * noise
                )

                noise_pred = model(x_t, t_norm)
                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B
                pbar.update(1)
                pbar.set_postfix(lr=scheduler.get_last_lr()[0])

        scheduler.step()

        if epoch % 50 == 0:
            sample_and_visualise(
                model,
                num_samples=num_samples,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                T=T,
                device=device,
                theta_list=theta_list,
                alpha_param=alpha_param,
                beta_param=beta_param,
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            }
            path = ckpt.format(step=epoch)
            torch.save(checkpoint, path)

        epoch_loss /= dataset_size * T
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6f}")


@torch.no_grad()
def sample(model, num_samples, betas, alphas, alpha_bars, T, device, theta_list):
    model.eval()
    samples_all = []
    for theta in theta_list:
        x = torch.randn(num_samples, 1, device=device, dtype=torch.float64)
        theta_tensor = torch.tensor(theta, device=device, dtype=torch.float64).view(
            1, -1
        )
        for t in reversed(range(1, T)):
            t_tensor = torch.full(
                (num_samples, 1), t / T, device=device, dtype=torch.float64
            )
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            alpha_bar_prev = alpha_bars[t - 1]

            shift_coeff = (1 - alpha_bar_t) / torch.sqrt(alpha_bar_t)
            theta_shifted = theta_tensor * shift_coeff
            shifted_x = x + theta_shifted

            eps = model(shifted_x, t_tensor)
            mask = shifted_x.abs() > THRESHOLD
            if mask.sum() > 0:
                print(f"Step {str(t).ljust(4)} | Entries > threshold: {mask.sum()}")

            score_if_large = -x / (1 - alpha_bar_t)
            score_if_small = -eps / torch.sqrt(1 - alpha_bar_t)
            score_p = torch.where(mask, score_if_large, score_if_small)
            score = score_p + theta_tensor / torch.sqrt(alpha_bar_t)

            z = torch.randn_like(x)
            x0 = (1 / torch.sqrt(alpha_bar_t)) * (x + score * (1 - alpha_bar_t))

            mu = (
                torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            ) * x + (torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)) * x0
            noise = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * z
            x = mu + noise
        samples_all.append(x.cpu().numpy())
    return samples_all


def _expected_mean_variance_twisted_beta(
    theta: float, a: float, b: float
) -> Tuple[float, float]:
    """Compute expected mean and variance of Beta(a,b) twisted by exp(theta x) via numeric integration."""
    # Beta normalization constant
    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    x_vals = np.linspace(0, 1, 1000)
    fx = x_vals ** (a - 1) * (1 - x_vals) ** (b - 1) / B
    w = fx * np.exp(theta * x_vals)
    Z = np.trapezoid(w, x_vals)
    EX = np.trapezoid(x_vals * w, x_vals) / Z
    EX2 = np.trapezoid(x_vals**2 * w, x_vals) / Z
    return EX, EX2 - EX**2


def visualize_results(
    samples_list: List[np.ndarray],
    theta_list: List[float] | Tuple[float, ...],
    save_path: str = "diffusion_beta_samples.png",
    alpha_param=2.0,
    beta_param=5.0,
):
    if len(samples_list) != len(theta_list):
        raise ValueError("samples_list and theta_list must have the same length")

    cols = len(samples_list)
    rows = 1
    plt.figure(figsize=(6 * cols, 6 * rows))

    a, b = alpha_param, beta_param
    # Base Beta PDF normalization
    B_const = math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    for i, (samples, theta) in enumerate(zip(samples_list, theta_list)):
        samples = samples.flatten()
        sample_mean = samples.mean()
        sample_var = samples.var()

        expected_mean, expected_var = _expected_mean_variance_twisted_beta(theta, a, b)

        ax = plt.subplot(rows, cols, i + 1)
        ax.hist(samples, bins=100, alpha=0.7, density=True)

        x_vals = np.linspace(0, 1, 300)
        base_pdf = x_vals ** (a - 1) * (1 - x_vals) ** (b - 1) / B_const
        if abs(theta) < 1e-12:
            y_vals = base_pdf
            ref_label = f"Beta PDF (a={a}, b={b})"
        else:
            unnorm = base_pdf * np.exp(theta * x_vals)
            Z = np.trapezoid(unnorm, x_vals)
            y_vals = unnorm / Z
            ref_label = f"Twisted Beta (θ={theta})"

        ax.plot(x_vals, y_vals, label=ref_label)
        ax.axvline(expected_mean, color="red", label="Expected Mean")
        ax.axvline(sample_mean, linestyle="--", label="Sample Mean")

        ax.set_title(
            f"$\\theta$={theta:.2f}\n"
            f"$\\hat{{\mu}}$={sample_mean:.4f} | $\\mu$={expected_mean:.4f}\n"
            f"$\\hat{{\sigma}}^2$={sample_var:.4f} | $\\sigma^2$={expected_var:.4f}\n"
        )
        ax.set_xlabel("x")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def sample_and_visualise(
    model,
    num_samples,
    betas,
    alphas,
    alpha_bars,
    T,
    device,
    theta_list,
    alpha_param,
    beta_param,
    save_path="diffusion_samples.png",
):
    samples_list = sample(
        model, num_samples, betas, alphas, alpha_bars, T, device, theta_list
    )
    visualize_results(samples_list, theta_list, save_path, alpha_param, beta_param)
    means = [s.mean() for s in samples_list]
    return means


def main():
    global THRESHOLD

    TRAIN = 0
    step = 1000

    T = 1000
    beta_start = 0.0001
    beta_end = 0.013

    THRESHOLD = 1e10
    theta_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    # theta_vals = [0]
    num_samples = 10000

    alpha_param = 2.0
    beta_param = 3.0
    num_points = 100000
    batch_size = 10000
    num_epochs = 1000
    learning_rate = 1e-3

    ckpt = f"model_beta({int(alpha_param)},{int(beta_param)})_{{step}}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    dist = create_beta_distribution(num_points, alpha_param, beta_param)
    model = ScoreNet(input_dim=1, dtype=torch.float64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    if step:
        print(f"Loading model from {ckpt.format(step=step)}")
        path = ckpt.format(step=step)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if TRAIN:
        dataset = TensorDataset(dist)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train(
            model,
            dataloader,
            optimizer,
            scheduler,
            betas,
            alphas,
            alpha_bars,
            T,
            device,
            alpha_param,
            beta_param,
            theta_list=theta_vals,
            ckpt=ckpt,
            num_epochs=num_epochs,
            num_samples=int(num_samples / (len(theta_vals) * 2)),
            start_epoch=step,
        )

    model.eval()
    sample_and_visualise(
        model,
        num_samples,
        betas,
        alphas,
        alpha_bars,
        T,
        device,
        theta_vals,
        alpha_param,
        beta_param,
    )


if __name__ == "__main__":
    main()
