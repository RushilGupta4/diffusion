import torch
import matplotlib.pyplot as plt
import numpy as np
import math


def get_diffusion_hyperparams(T, beta_start, beta_end, device):
    """Creates the beta schedule and corresponding cumulative products."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


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


def visualize_results(samples, a, b, theta, save_path="diffusion_samples.png"):
    """Plots a histogram of the generated samples and overlays the true Uniform(a, b) density."""
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    samples = samples.flatten()
    sample_mean = samples.mean()
    sample_var = samples.var()  # Calculate sample variance
    expected_mean = _expected_mean_twisted(theta, a, b)  # Calculate expected mean
    expected_var = _expected_variance_twisted(theta, a, b)  # Calculate expected variance

    e = (b - a) / 5

    # Ensure the samples are within the range [a - e, b + e] (by removing outliers)
    num_samples = len(samples)
    samples = samples[(samples >= a - e) & (samples <= b + e)]
    num_new_samples = len(samples)

    print("Removed outliers: ", num_samples - num_new_samples)

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=100, alpha=0.7, density=True, label="Samples")

    # Uniform PDF on [a, b]
    x = np.linspace(a, b, 5000)
    if abs(theta) < 1e-12:
        y = np.ones_like(x) / (b - a)
        dist_label = f"Uniform[{a}, {b}] PDF"
    else:
        Z = (np.exp(theta * b) - np.exp(theta * a)) / theta
        y = np.exp(theta * x) / Z
        dist_label = f"Twisted Uniform[{a}, {b}] PDF ($\\theta={theta:.2f}$)"

    plt.plot(x, y, color="red", label=dist_label)

    # Add vertical lines for means
    plt.axvline(expected_mean, color="red", linestyle="-")
    plt.axvline(sample_mean, color="blue", linestyle="--")

    # Decorations
    # Update title to include mean and variance info
    plt.title(
        f"Histogram of Generated Samples\n"
        f"Sample: $\\hat{{\\mu}}={sample_mean:.4f}$, $\\hat{{\\sigma}}^2={sample_var:.4f}$\n"
        f"Expected: $\\mu={expected_mean:.4f}$, $\\sigma^2={expected_var:.4f}$"
    )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.xlim(a - e, b + e)
    plt.savefig(save_path)
    plt.close()


def score_func(start, end, x, alpha_bar_t):
    """
    Calculates the score grad_x log p_t(x) for the VP-SDE forward process:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * z
    where x_0 ~ Uniform(start, end).
    """

    sigma_t = torch.sqrt(1 - alpha_bar_t)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)

    if alpha_bar_t < 0.01:
        return -x - sqrt_alpha_bar_t * (start + end) / 2

    effective_start = sqrt_alpha_bar_t * start
    effective_end = sqrt_alpha_bar_t * end

    z_start = (x - effective_start) / sigma_t
    z_end = (x - effective_end) / sigma_t

    log_pdf_start = -0.5 * z_start**2 - 0.5 * math.log(2 * math.pi)
    log_pdf_end = -0.5 * z_end**2 - 0.5 * math.log(2 * math.pi)
    pdf_start = torch.exp(log_pdf_start)
    pdf_end = torch.exp(log_pdf_end)

    cdf_start = 0.5 * (1 + torch.erf(z_start / math.sqrt(2)))
    cdf_end = 0.5 * (1 + torch.erf(z_end / math.sqrt(2)))

    denominator = sigma_t * (cdf_start - cdf_end)
    epsilon = 1e-12

    score = (pdf_start - pdf_end) / (
        denominator + torch.sign(denominator) * epsilon + epsilon
    )

    high_mask = (x > end)
    stable_high_score = -z_end / sigma_t
    score[high_mask] = stable_high_score[high_mask]

    low_mask = (x < start)
    stable_low_score = -z_start / sigma_t
    score[low_mask] = stable_low_score[low_mask]

    return score


def sample(num_samples, start, end, betas, alphas, alpha_bars, T, device, theta):
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
            score_p = score_func(start, end, shifted_x, alpha_bar_t_val)
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
    T = 1000  # Total number of diffusion steps
    beta_start = 0.0001
    beta_end = 0.014

    num_samples = 100000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas, alphas, alpha_bars = get_diffusion_hyperparams(
        T, beta_start, beta_end, device
    )

    # Set theta to a nonzero value to introduce drift.
    theta = -1
    theta_float = float(theta)  # Use float for calculations
    theta_vector = torch.tensor([theta_float], device=device, dtype=torch.float32)  # Use tensor for sampling

    start = -5
    end = 5

    # Generate samples using the reverse process.
    samples = sample(
        num_samples=num_samples,
        start=start,
        end=end,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=T,
        device=device,
        theta=theta_vector,  # Pass tensor to sample
    )

    visualize_results(samples, start, end, theta_float)  # Pass float theta to visualize


if __name__ == "__main__":
    main()
