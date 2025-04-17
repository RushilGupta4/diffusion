import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


class ScoreNet(nn.Module):
    """Feed-forward score network for the diffusion model."""

    def __init__(
        self,
        input_dim=1,
        time_embedding=16,
        theta_embedding=16,
        hidden_dim=128,
        num_layers=3,
    ):
        super(ScoreNet, self).__init__()
        layers = [
            nn.Linear(input_dim + theta_embedding + time_embedding, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

        self.time_net = nn.Sequential(
            nn.Linear(1, time_embedding),
            nn.ReLU(),
            nn.Linear(time_embedding, time_embedding),
            nn.ReLU(),
            nn.Linear(time_embedding, time_embedding),
        )

        self.theta_net = nn.Sequential(
            nn.Linear(1, theta_embedding),
            nn.ReLU(),
            nn.Linear(theta_embedding, theta_embedding),
            nn.ReLU(),
            nn.Linear(theta_embedding, theta_embedding),
        )

        print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x, t, theta):
        # Embed time and theta, then concatenate along the feature dimension (batch x (input_dim + time_embedding + theta_embedding))
        t = self.time_net(t)
        theta = self.theta_net(theta)
        x = torch.cat([x, t, theta], dim=1)
        return self.net(x)


def get_diffusion_hyperparams(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def create_normal_distribution(num_points=1000, mean=0.0, std=1.0):
    data = np.random.normal(mean, std, num_points)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(1)


def train_diffusion_weighted(
    model,
    data,
    theta_values,
    optimizer,
    scheduler,
    betas,
    alphas,
    alpha_bars,
    T,
    device,
    num_epochs=300,
    batch_size=128,
    ckpt="model_{step}.pth",
    start_epoch=0,
):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Compute a global maximum weight over all samples and theta values for normalization.
    weight_map = {}
    for th in theta_values:
        w = torch.exp(data * th)
        weight_map[th] = w.max().item()

    dataset_size = len(dataset)
    num_batches = len(dataloader)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_loss = 0.0
        total_samples = 0

        with tqdm(total=num_batches, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for (x0,) in dataloader:
                x0 = x0.to(device)
                B = x0.size(0)
                total_samples += B

                theta_index = np.random.randint(0, len(theta_values))
                theta_sample = theta_values[theta_index]
                theta_tensor = torch.full((B, 1), float(theta_sample), device=device)

                t = torch.randint(0, T, (B,), device=device)
                t_norm = (t.float() / T).unsqueeze(1)

                noise = torch.randn_like(x0)
                sqrt_alpha_bar = torch.sqrt(alpha_bars[t]).unsqueeze(1)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t]).unsqueeze(1)

                x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

                noise_pred = model(x_t, t_norm, theta_tensor)

                weights = torch.exp(theta_tensor * x0)
                loss = (
                    weights * ((noise_pred - noise) ** 2) / weight_map[theta_sample]
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B
                avg_loss = epoch_loss / total_samples

                pbar.update(1)
                pbar.set_postfix(loss=avg_loss, lr=scheduler.get_last_lr()[0])

        scheduler.step()
        if epoch % 50 != 0:
            continue

        samples_list = []
        for theta in theta_values:
            samples = sample(
                model,
                num_samples=5000,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                T=T,
                device=device,
                theta=theta,
            )
            samples_list.append(samples)

        visualize_results(samples_list=samples_list, theta_list=theta_values)

        mean = samples.mean()
        path = ckpt.format(step=epoch)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            path,
        )

        epoch_loss /= dataset_size
        print(
            f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6f}, Mean: ({mean:.4f})\n\n"
        )

    return model


@torch.no_grad()
def sample(model, num_samples, betas, alphas, alpha_bars, T, device, theta: float):
    model.eval()
    x = torch.randn(num_samples, 1, device=device)
    theta_val = torch.ones_like(x) * theta

    for t in reversed(range(1, T)):
        t_tensor = torch.full((num_samples, 1), t / T, device=device)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1]

        eps = model(x, t_tensor, theta_val)
        score = -eps / torch.sqrt(1 - alpha_bar_t)

        if t > 1:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x0 = (1 / torch.sqrt(alpha_bar_t)) * (x + score * (1 - alpha_bar_t))

        mu = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * x + (
            torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        ) * x0
        noise = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * z
        x = mu + noise

    return x.cpu().numpy()


def visualize_results(
    samples_list,
    theta_list,
    save_path="diffusion_samples.png",
    base_mean=4,
):
    num_plots = len(samples_list)
    cols = num_plots
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (samples, theta) in enumerate(zip(samples_list, theta_list)):
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

        samples = samples.flatten()
        sample_mean = samples.mean()

        ax = axes[i]
        ax.hist(samples, bins=100, alpha=0.7, density=True)

        true_mean = base_mean + theta
        x_vals = np.linspace(true_mean - 5, true_mean + 5, 200)
        y_vals = (1 / (np.sqrt(2 * np.pi) * 1)) * np.exp(
            -0.5 * ((x_vals - true_mean) ** 2)
        )
        ax.plot(x_vals, y_vals, label="True Distribution", color="red")
        ax.axvline(sample_mean, color="blue", label="Sample Mean")
        ax.axvline(true_mean, color="red", label="True Mean")

        ax.set_title(f"$\\theta$ = {theta:.2f}\nSample $\\mu$ = {sample_mean:.4f}")
        ax.set_xlim(true_mean - 5, true_mean + 5)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    TRAIN = 1
    step = 0
    ckpt = "model1_{step}.pth"

    T = 1000  # Total number of diffusion steps
    beta_start = 0.0001
    beta_end = 0.02

    num_samples = 15000
    num_points = 100000
    batch_size = 10000

    num_epochs = 500
    learning_rate = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    # Create the base dataset (normal distribution with mean=4).
    dist = create_normal_distribution(num_points=num_points, mean=4, std=1.0)

    # Define a set of theta values.
    theta_values = np.linspace(-1, 1, 5)
    print("theta_values:", theta_values.tolist())

    model = ScoreNet(input_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    if step:
        # Load the trained model and optimizer state
        print(f"Loading model from {ckpt.format(step=step)}")
        path = ckpt.format(step=step)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if TRAIN:
        train_diffusion_weighted(
            model,
            dist,
            theta_values,
            optimizer,
            scheduler,
            betas,
            alphas,
            alpha_bars,
            T,
            device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            ckpt=ckpt,
            start_epoch=step,
        )

    samples_list = []
    for theta in theta_values:
        samples = sample(
            model,
            num_samples=num_samples,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            T=T,
            device=device,
            theta=theta,
        )
        samples_list.append(samples)
        print(f"For theta = {theta:.3f}, sample mean: {samples.mean():.4f}")

    visualize_results(samples_list, theta_list=theta_values)


if __name__ == "__main__":
    main()
