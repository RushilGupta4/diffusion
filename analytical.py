import torch
import matplotlib.pyplot as plt
import numpy as np


mean = 4


def get_diffusion_hyperparams(T, beta_start, beta_end, device):
    """Creates the beta schedule and corresponding cumulative products."""
    # Use a linear schedule for betas.
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def visualize_results(samples, save_path="diffusion_samples.png"):
    """Plots a histogram of the generated samples."""
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    samples = samples.flatten()
    sample_mean = samples.mean()

    x = np.linspace(samples.min(), samples.max(), 5000)
    y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) ** 2))

    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=100, alpha=0.7, density=True)
    plt.plot(x, y, color="red", label="True Distribution")
    plt.title("Histogram of Generated Samples\nMean: {:.4f}".format(sample_mean))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()


def score_func(mu, x, alpha_bar_t):
    return mu * torch.sqrt(alpha_bar_t) - x


def sample(num_samples, mu_dist, betas, alphas, alpha_bars, T, device, theta):
    x = torch.randn(num_samples, 1, device=device)

    # Ensure theta is a tensor and reshape for broadcasting.
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, device=device, dtype=torch.float32)
    theta = theta.view(1, -1)

    # Use no_grad to prevent unnecessary gradient computation.
    with torch.no_grad():
        for t in reversed(range(1, T)):
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t_val = alpha_bars[t]
            alpha_bar_prev = alpha_bars[t - 1]

            shift = theta * (1 - alpha_bar_t_val) / torch.sqrt(alpha_bar_t_val)
            shifted_x = x + shift
            # score_p = empirical_score(shifted_x, data, alpha_bar_t_val)
            score_p = score_func(mu_dist, shifted_x, alpha_bar_t_val)
            score = score_p + theta / torch.sqrt(alpha_bar_t_val)

            z = torch.randn_like(x)
            x0 = (1 / torch.sqrt(alpha_bar_t_val)) * (x + score * (1 - alpha_bar_t_val))

            mu = (
                torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t_val)
            ) * x + (torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t_val)) * x0
            noise = (
                torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t_val)) * z
            )
            x = mu + noise

    return x.cpu().numpy()


def main():
    # Hyperparameters
    global mean

    T = 2000  # Total number of diffusion steps
    beta_start = 0.0001
    beta_end = 0.02

    num_samples = 50000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    # Set theta to a nonzero value to introduce drift.
    theta = 2
    theta_vector = np.array([1]) * theta

    mu_dist = 5
    mean = mu_dist + theta

    # Generate samples using the reverse process.
    samples = sample(
        num_samples=num_samples,
        mu_dist=mu_dist,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=T,
        device=device,
        theta=theta_vector,
    )

    visualize_results(samples)


if __name__ == "__main__":
    main()
