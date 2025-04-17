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


def create_uniform_distribution(num_points=1000, start=0.0, end=1.0):
    """Generates a uniform distribution dataset between start and end."""
    data = np.random.uniform(start, end, num_points)
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
    start=0.0,
    end=1.0,
    theta_list=[0],
    ckpt="model_{step}.pth",
    start_epoch=0,
    num_epochs=10,
    num_samples=50000,
):
    """Training loop for the diffusion model with uniform data."""

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_loss = 0.0
        total_steps = num_batches

        with tqdm(total=total_steps, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for (x0,) in dataloader:
                x0 = x0.to(device)
                B = x0.shape[0]

                t = torch.randint(0, T, (B,), device=device)
                t_norm = t.float().to(torch.float64) / T
                t_norm = t_norm.unsqueeze(1)

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
                base_start=start,
                base_end=end,
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


def _expected_mean_twisted(theta: float, start: float, end: float) -> float:
    a = start
    b = end

    if abs(theta) < 1e-12:
        return 0.5 * (start + end)

    exp_a = np.exp(theta * a)
    exp_b = np.exp(theta * b)

    Z = (exp_b - exp_a) / theta
    dZ = (b * exp_b - a * exp_a) / theta - (exp_b - exp_a) / theta**2

    return dZ / Z


def _expected_variance_twisted(theta: float, start: float, end: float) -> float:
    a, b = start, end

    if abs(theta) < 1e-12:
        return (b - a) ** 2 / 12.0

    exp_a, exp_b = np.exp(theta * a), np.exp(theta * b)
    Z = (exp_b - exp_a) / theta

    m1_num = np.exp(theta * b) * (b / theta - 1 / theta**2) - np.exp(theta * a) * (
        a / theta - 1 / theta**2
    )
    m2_num = np.exp(theta * b) * (
        b**2 / theta - 2 * b / theta**2 + 2 / theta**3
    ) - np.exp(theta * a) * (a**2 / theta - 2 * a / theta**2 + 2 / theta**3)

    EX = m1_num / Z
    EX2 = m2_num / Z

    return EX2 - EX**2


def visualize_results(
    samples_list: List[np.ndarray],
    theta_list: List[float] | Tuple[float, ...],
    save_path: str = "diffusion_samples.png",
    base_start: float = 0.0,
    base_end: float = 1.0,
):
    if len(samples_list) != len(theta_list):
        raise ValueError("samples_list and theta_list must have the same length")

    num_plots = len(samples_list)

    # Determine a near‑square grid layout
    cols = num_plots
    rows = 1

    print(f"Rows: {rows}, Cols: {cols}")

    # Each cell gets a 6×6 inch canvas
    plt.figure(figsize=(6 * cols, 6 * rows))

    for i, (samples, theta) in enumerate(zip(samples_list, theta_list)):
        samples = samples.flatten()
        sample_mean = samples.mean()
        sample_var = samples.var()

        expected_mean = _expected_mean_twisted(theta, base_start, base_end)
        expected_var = _expected_variance_twisted(theta, base_start, base_end)

        ax = plt.subplot(rows, cols, i + 1)
        ax.hist(samples, bins=100, alpha=0.7, density=True)

        x_vals = np.linspace(base_start, base_end, 300)
        if abs(theta) < 1e-12:
            y_vals = np.ones_like(x_vals) / (base_end - base_start)
            ref_label = "Uniform Density"
        else:
            Z = (np.exp(theta * base_end) - np.exp(theta * base_start)) / theta
            y_vals = np.exp(theta * x_vals) / Z
            ref_label = f"Twisted Density (θ={theta})"

        ax.plot(x_vals, y_vals, label=ref_label)
        ax.axvline(expected_mean, color="red", label="Expected Mean")
        ax.axvline(sample_mean, color="blue", linestyle="--", label="Sample Mean")

        ax.set_title(
            f"$\\theta = {theta:.2f}$"
            f"\n$\\hat\\mu = {sample_mean:.4f}$ | $\\mu = {expected_mean:.4f}$"
            f"\n$\\hat\\sigma^2 = {sample_var:.4f}$ | $\\sigma^2 = {expected_var:.4f}$"
        )
        ax.set_xlabel("Value")
        ax.set_xlim(base_start - 0.5, base_end + 0.5)
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
    save_path="diffusion_samples.png",
    base_start=0.0,
    base_end=1.0,
):
    samples_list = sample(
        model, num_samples, betas, alphas, alpha_bars, T, device, theta_list
    )
    visualize_results(samples_list, theta_list, save_path, base_start, base_end)
    means = [s.mean() for s in samples_list]
    return means


def main():
    global THRESHOLD

    TRAIN = 0
    step = 500
    ckpt = "model_{step}.pth"

    T = 1000
    beta_start = 0.0001
    beta_end = 0.013

    THRESHOLD = 10000000000
    theta_vals = [-0.5, -0.25, 0, 0.25, 0.5]
    num_samples = 10000

    start = -2.5
    end = 2.5

    num_points = 100000
    batch_size = 10000
    num_epochs = 500
    learning_rate = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    print(f"Alpha Bar Last: {alpha_bars[-1].item()}")
    print(
        f"Theta Scale Last: {((1 - alpha_bars[-1]) / (alpha_bars[-1] ** 0.5)).item()}\n"
    )

    dist = create_uniform_distribution(num_points=num_points, start=start, end=end)

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
            start=start,
            end=end,
            theta_list=theta_vals,
            ckpt=ckpt,
            num_epochs=num_epochs,
            num_samples=int(num_samples / len(theta_vals)),
            start_epoch=step,
        )
    else:
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
        base_start=start,
        base_end=end,
    )


if __name__ == "__main__":
    main()
