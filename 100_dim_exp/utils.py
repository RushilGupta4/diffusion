import torch

import matplotlib.pyplot as plt
from model import ScoreNet
from tqdm import tqdm


import numpy as np
from typing import Tuple


def linear_beta_schedule(T, beta_start, beta_end, device, dtype):
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def scaled_linear_beta_schedule(T, beta_start, beta_end, device, dtype):
    betas = (
        torch.linspace(beta_start**0.5, beta_end**0.5, T, dtype=dtype, device=device)
        ** 2
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def get_score(model, x, theta, t, alpha_bar_t, num_samples, bounded=False):
    threshold = 0.01

    if alpha_bar_t < threshold:
        score = -x / (1 - alpha_bar_t)

    else:
        sqrt_a = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_a = torch.sqrt(1 - alpha_bar_t)

        theta = theta.expand(num_samples, x.shape[1])
        shift_coeff = (1 - alpha_bar_t) / sqrt_a
        shifted_x = x + theta * shift_coeff

        score = -model(shifted_x, t) / sqrt_one_minus_a

        offset = 0.5
        mask_large = bounded & (shifted_x > 1 + offset)
        mask_small = bounded & (shifted_x < -offset)

        score[mask_large] = ((sqrt_a - shifted_x) / (1 - alpha_bar_t))[mask_large]
        score[mask_small] = (-shifted_x / (1 - alpha_bar_t))[mask_small]

        score = score + theta / torch.sqrt(alpha_bar_t)

    return score


@torch.no_grad()
def ddpm_sample(
    model: ScoreNet,
    num_samples,
    betas,
    alphas,
    alpha_bars,
    T,
    device,
    theta,
    dim,
    bounded=False,
):
    model.eval()

    # if type of theta is not torch.Tensor, convert it to torch.Tensor
    if not isinstance(theta, torch.Tensor):
        theta_tensor = torch.tensor(theta, device=device, dtype=model.dtype)
    else:
        theta_tensor = theta.to(device, dtype=model.dtype)

    x = torch.randn(num_samples, dim, device=device, dtype=model.dtype)

    with tqdm(total=T, desc="DDPM Sampling") as pbar:

        for t in reversed(range(1, T)):
            t_tensor = torch.full(
                (num_samples, 1), t / T, device=device, dtype=model.dtype
            )
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            alpha_bar_prev = alpha_bars[t - 1]

            score = get_score(
                model, x, theta_tensor, t_tensor, alpha_bar_t, num_samples, bounded
            )

            z = torch.randn_like(x)
            x0 = (1 / torch.sqrt(alpha_bar_t)) * (x + score * (1 - alpha_bar_t))

            mu = (
                torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            ) * x + (torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)) * x0
            noise = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * z
            x = mu + noise

            pbar.update(1)

    return x.cpu().numpy()


def plot_losses(losses, path):
    # Get the number of loss types
    num_losses = len(losses[0].keys())

    # Create a grid of subplots (adjust rows and columns based on the number of losses)
    if num_losses <= 3:
        nrows, ncols = num_losses, 1  # One column if 3 or fewer losses
    else:
        nrows = (num_losses + 1) // 2  # Ceiling division
        ncols = 2

    # Create a new figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))

    # Make axes iterable if there's only one subplot
    if num_losses == 1:
        axes = [axes]
    # Make axes iterable if there's only one row
    elif nrows == 1:
        axes = [axes]
    # Flatten the 2D array of axes for easy iteration
    elif nrows > 1 and ncols > 1:
        axes = axes.flatten()

    # Iterate through the losses and plot each one in its own subplot
    for i, key in enumerate(losses[0].keys()):
        ax = axes[i]
        ax.plot([loss[key] for loss in losses])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{key} Loss")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.7)

    # Hide any unused subplots
    for i in range(num_losses, nrows * ncols):
        if i < len(axes):
            axes[i].set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(path)
    plt.close()


def find_theta(
    dist: np.ndarray,
    threshold: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000,
    bracket: Tuple[float, float] = (-20.0, 20.0),
) -> np.ndarray:
    x = dist.astype(np.float64)  # (N, d)
    s = x.sum(axis=1)  # (N,)  — Σ_j x_j for each sample
    d = x.shape[1]

    # Helper: expectation of Σx under Q_λ
    def moment(lam: float) -> float:
        z = lam * s
        z -= z.max()  # log-sum-exp stabilisation
        w = np.exp(z)
        Z = w.sum()
        return np.dot(w, s) / Z  # E_{Q_λ}[Σx]

    # If the raw mean already meets / exceeds the threshold,
    # λ = 0 is optimal.
    base_mean = s.mean()
    if abs(base_mean - threshold) <= tol:
        return np.zeros(d, dtype=np.float64)

    # Ensure the bracket contains a root.
    lo, hi = bracket
    lo_m, hi_m = moment(lo), moment(hi)

    # Expand the bracket if necessary.
    expand = 2.0
    n_expand = 0
    while (lo_m - threshold) * (hi_m - threshold) > 0 and n_expand < 20:
        if base_mean < threshold:
            hi *= expand
            hi_m = moment(hi)
        else:
            lo *= expand
            lo_m = moment(lo)
        n_expand += 1

    if (lo_m - threshold) * (hi_m - threshold) > 0:
        raise RuntimeError("Could not bracket the root for λ.")

    # Bisection.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_m = moment(mid)
        if abs(mid_m - threshold) <= tol:
            lam = mid
            break
        if (lo_m - threshold) * (mid_m - threshold) < 0:
            hi, hi_m = mid, mid_m
        else:
            lo, lo_m = mid, mid_m
    else:
        raise RuntimeError("Bisection did not converge.")

    lam = 0.5 * (lo + hi)
    theta = np.full(d, lam, dtype=np.float64)
    return theta


def plot_comparison(dist1, dist2, title1, title2, path):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: histogram of dist1 (dist1 distribution)
    sum_dist1 = np.sum(dist1, axis=1)
    mean1, std1 = np.mean(sum_dist1), np.std(sum_dist1)

    axes[0].hist(sum_dist1, bins=100, density=True)
    axes[0].set_title(f"{title1}\n$\mu={mean1:.4f}, \sigma={std1:.4f}$")
    axes[0].set_xlabel("Sum of samples")
    axes[0].set_ylabel("Density")

    # Second subplot: histogram of samples (generated samples)
    sum_dist2 = np.sum(dist2, axis=1)
    mean2, std2 = np.mean(sum_dist2), np.std(sum_dist2)

    axes[1].hist(sum_dist2, bins=100, density=True)
    axes[1].set_title(f"{title2}\n$\mu={mean2:.4f}, \sigma={std2:.4f}$")
    axes[1].set_xlabel("Sum of samples")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
