import os
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import ScoreNet
from utils import (
    scaled_linear_beta_schedule,
    ddpm_sample,
    plot_losses,
    find_theta,
    plot_comparison,
)
from dist import (
    create_multivariate_normal,
    compare_dist,
    rejection_sampling,
    create_multivariate_beta_using_copula,
)


sample = ddpm_sample


def train(
    model: ScoreNet,
    losses,
    dataloader,
    dist,
    optimizer,
    scheduler,
    betas,
    alphas,
    alpha_bars,
    num_samples,
    theta,
    T,
    device,
    ckpt="model_{step}.pth",
    png_file="comparison.png",
    losses_file="losses.png",
    start_epoch=0,
    num_epochs=10,
    dim=2,
    bounded=False,
):
    """Training loop for the diffusion model with uniform data."""
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_loss = 0.0
        total_steps = num_batches

        with tqdm(total=total_steps, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for x0, weights in dataloader:
                x0 = x0.to(device)
                weights = weights.to(device)
                B = x0.shape[0]

                t = torch.randint(0, T, (B,), device=device)
                t_norm = t.float().to(model.dtype) / T
                t_norm = t_norm.unsqueeze(1)

                noise = torch.randn_like(x0)
                x_t = (
                    torch.sqrt(alpha_bars[t]).unsqueeze(1) * x0
                    + torch.sqrt(1 - alpha_bars[t]).unsqueeze(1) * noise
                )

                noise_pred = model(x_t, t_norm)
                loss = (weights * ((noise_pred - noise) ** 2)).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item() * B
                pbar.update(1)
                pbar.set_postfix(lr=scheduler.get_last_lr()[0])

        scheduler.step()

        if epoch == num_epochs or epoch % 25 == 0:
            theta_zero = torch.zeros_like(theta)
            samples = sample(
                model,
                num_samples,
                betas,
                alphas,
                alpha_bars,
                T,
                device,
                theta=theta_zero,
                dim=dim,
                bounded=bounded,
            )

            # Compare with input distribution using rejection sampling
            inp_dist = rejection_sampling(
                dist.cpu().numpy(), num_samples, theta=theta.cpu().numpy()
            )
            diff, distance = compare_dist(samples, inp_dist)
            for key, value in distance.items():
                print(f"{key}: {value:.4f}", end=" | ")
            print()

            path_folder = ckpt.split("/")
            if len(path_folder) > 1:
                path_folder = "/".join(path_folder[:-1])
                if not os.path.exists(path_folder):
                    os.makedirs(path_folder)

            plot_comparison(
                inp_dist,
                samples,
                "Rejection Sampling Distribution",
                "Diffusion Model Distribution",
                path=png_file,
            )

            losses.append(distance)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "losses": losses,
            }
            path = ckpt.format(step=epoch)
            torch.save(checkpoint, path)

            plot_losses(
                losses,
                path=losses_file,
            )

        epoch_loss /= dataset_size * T
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.8f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded")
    args = parser.parse_args()

    if args.bounded:
        BOUNDED = True
    else:
        BOUNDED = False

    TRAIN = 0

    if BOUNDED:
        ckpt = "model_weighted_bounded/epoch_{step}.pth"
        png_file = "model_weighted_bounded/comparison.png"
        losses_file = "model_weighted_bounded/losses.png"
    else:
        ckpt = "model_weighted/epoch_{step}.pth"
        png_file = "model_weighted/comparison.png"
        losses_file = "model_weighted/losses.png"

    step = 200
    num_epochs = 500

    T = 1000
    beta_start = 0.00085
    beta_end = 0.012

    num_samples = 250_000
    num_samples = 1_000_000
    num_points = 5_000_000 if TRAIN else num_samples * 5
    batch_size = 50_000

    learning_rate = 1e-4
    dim = 100
    dtype = torch.float64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(dtype)

    betas, alphas, alpha_bars = scaled_linear_beta_schedule(
        T, beta_start, beta_end, device, dtype
    )

    if BOUNDED:
        dist = create_multivariate_beta_using_copula(
            num_points=num_points, dim=dim, dtype=dtype
        ).to(device)
    else:
        dist = create_multivariate_normal(
            num_points=num_points, dtype=dtype, dim=dim
        ).to(device)

    threshold = 50.5 if BOUNDED else 28
    theta = find_theta(dist.cpu().numpy(), threshold)
    theta = torch.tensor(theta, dtype=dtype, device=device)
    theta = (0.41613638401031494 if BOUNDED else 0.24622570723295212) * torch.ones_like(
        theta
    )
    weights = torch.exp(torch.sum(theta.view(1, -1) * dist, dim=1, keepdim=True))

    print(theta.max().item())
    print((theta**2).sum().sqrt().item())
    model = ScoreNet(input_dim=dim, dtype=dtype, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-6
    )
    losses = []

    if step:
        print(f"Loading model from {ckpt.format(step=step)}")
        path = ckpt.format(step=step)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        losses = checkpoint["losses"]

    if TRAIN:
        dataset = TensorDataset(dist, weights)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train(
            model,
            losses,
            dataloader,
            dist,
            optimizer,
            scheduler,
            betas,
            alphas,
            alpha_bars,
            num_samples,
            theta,
            T,
            device,
            ckpt=ckpt,
            png_file=png_file,
            losses_file=losses_file,
            num_epochs=num_epochs,
            start_epoch=step,
            dim=dim,
            bounded=BOUNDED,
        )

    model.eval()

    theta_zero = torch.zeros_like(theta)
    samples = sample(
        model,
        num_samples,
        betas,
        alphas,
        alpha_bars,
        T,
        device,
        theta=theta_zero,
        dim=dim,
        bounded=BOUNDED,
    )

    theta = theta.cpu().numpy()
    dist = dist.cpu().numpy()

    inp_dist = rejection_sampling(dist, num_samples, theta=theta)
    diff, distance = compare_dist(samples, inp_dist)
    for key, value in distance.items():
        print(f"{key}: {value:.4f}", end=" | ")

    print()

    if not os.path.exists(os.path.dirname(png_file)):
        os.makedirs(os.path.dirname(png_file))

    plot_comparison(
        inp_dist,
        samples,
        "Rejection Sampling Distribution",
        "Diffusion Model Distribution",
        path=png_file,
    )


if __name__ == "__main__":
    main()
