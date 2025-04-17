import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

THRESHOLD = 10


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


def create_normal_distribution(num_points=1000, mean=0.0, std=1.0):
    """Generates a normal distribution dataset."""
    data = np.random.normal(mean, std, num_points)
    return torch.tensor(data, dtype=torch.float64).unsqueeze(1)  # Add a dimension


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
    start_epoch=0,
    num_epochs=10,
    num_samples=5000,
    theta_list=[0],
    true_mean=4,
    std=1.0,
    ckpt="model_{step}.pth",
):
    """Training loop for the diffusion model with a single tqdm progress bar per epoch."""
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        total_steps = num_batches

        with tqdm(total=total_steps, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for (x0,) in dataloader:
                x0 = x0.to(device)
                B = x0.shape[0]

                t = torch.randint(0, T, (B,), device=device)
                t_norm = t.float().to(torch.float64) / T
                t_norm = t_norm.unsqueeze(1)

                sqrt_alpha_bar_t = torch.sqrt(alpha_bars[t]).unsqueeze(1)
                sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t]).unsqueeze(1)

                noise = torch.randn_like(x0)
                x_t = sqrt_alpha_bar_t * x0 + sqrt_1_minus_alpha_bar_t * noise

                noise_pred = model(x_t, t_norm)
                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B
                pbar.update(1)
                pbar.set_postfix(lr=scheduler.get_last_lr()[0])

        scheduler.step()

        if epoch % 25 != 0:
            continue

        means = sample_and_visualise(
            model,
            num_samples,
            betas,
            alphas,
            alpha_bars,
            T,
            device,
            theta_list=theta_list,
            base_mean=true_mean,
            base_std=std,
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
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Mean: ({', '.join([f'{m:.4f}' for m in means])})\n\n"
        )


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
                print(
                    f"Step {str(t).ljust(4)} | Number of entries that exceed the threshold: {mask.sum()}"
                )

            score_if_large = -x / (1 - alpha_bar_t)  # Gaussian score
            score_if_small = -eps / torch.sqrt(1 - alpha_bar_t)  # True score

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


def visualize_results(
    samples_list,
    theta_list,
    save_path="diffusion_samples.png",
    base_mean=4,
    base_std=1.0,
):
    num_plots = len(samples_list)
    plt.figure(figsize=(6 * num_plots, 6))
    for i, (samples, theta) in enumerate(zip(samples_list, theta_list)):
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

        samples = samples.flatten()
        sample_mean = samples.mean()
        plt.subplot(1, num_plots, i + 1)
        plt.hist(samples, bins=100, alpha=0.7, density=True)
        true_mean = base_mean + theta * (base_std**2)

        min_x = true_mean - 3 * base_std
        max_x = true_mean + 3 * base_std

        x_vals = np.linspace(min_x, max_x, 100)
        y_vals = (
            1
            / (np.sqrt(2 * np.pi * (base_std**2)))
            * np.exp(-0.5 * ((x_vals - true_mean) ** 2) / (base_std**2))
        )

        plt.plot(x_vals, y_vals, color="red", label="True Distribution")
        plt.axvline(sample_mean, color="blue", label="Sample Mean")
        plt.axvline(true_mean, color="red", label="True Mean")

        plt.title(
            "Theta: {:.2f}\nSample Mean: {:.4f}\nTrue Mean: {:.4f}".format(
                theta, sample_mean, true_mean
            )
        )

        plt.xlabel("Value")
        plt.xlim(min_x, max_x)
        plt.ylabel("Frequency")
        plt.legend()

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
    base_mean=4,
    base_std=1.0,
):
    samples_list = sample(
        model, num_samples, betas, alphas, alpha_bars, T, device, theta_list
    )
    visualize_results(samples_list, theta_list, save_path, base_mean, base_std)
    means = [s.mean() for s in samples_list]
    return means


def main():
    global THRESHOLD

    TRAIN = 0
    step = 200
    ckpt = "model_{step}.pth"

    # Hyperparameters
    T = 1000  # total number of diffusion steps
    beta_start = 0.0001
    beta_end = 0.014

    THRESHOLD = 100
    theta_vals = [0, 0.2]
    num_samples = 50000
    num_samples = 10000

    true_mean = 0
    std = np.sqrt(2)

    num_points = 100000
    batch_size = 10000

    num_epochs = 251
    learning_rate = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default tensor type to float64
    torch.set_default_dtype(torch.float64)

    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    print(f"Alpha Bar Last: {alpha_bars[-1].item()}")
    print(
        f"Theta Scale Last: {((1 - alpha_bars[-1]) / ((alpha_bars[-1]) ** 0.5)).item()}"
    )
    print()

    dist = create_normal_distribution(num_points=num_points, mean=true_mean, std=std)

    model = ScoreNet(input_dim=1, dtype=torch.float64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Create scheduler and load its state
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
        # Create the dataloader
        dataset = TensorDataset(dist)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train the model
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
            start_epoch=step + 1,
            num_epochs=num_epochs,
            ckpt=ckpt,
            num_samples=int(num_samples / 2),
            theta_list=theta_vals,
            true_mean=true_mean,
            std=std,
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
        base_mean=true_mean,
        base_std=std,
    )


if __name__ == "__main__":
    main()
