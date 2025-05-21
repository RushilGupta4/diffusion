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
    create_twisted_multivariate_normal
)


sample = ddpm_sample

TARGET_THETA = None  # This is initialised in __main__
THETA_STEPS = 2
BOUNDED = False  # This is initialised in __main__
dtype = torch.float64


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

        epoch_loss /= dataset_size * T
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.8f}")


        if epoch == num_epochs or epoch % 50 == 0:
            theta_zero = torch.zeros_like(theta)
            batch_size = dataset_size // num_batches
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
                batch_size=batch_size*2
            )

            # Compare with input distribution using rejection sampling
            if BOUNDED:
                inp_dist = rejection_sampling(
                    dist.cpu().numpy(), num_samples, theta=theta.cpu().numpy()
                )
            else:
                inp_dist = create_twisted_multivariate_normal(
                    theta=theta.cpu().numpy(),
                    num_points=num_samples,
                    dim=dim,
                    dtype=dtype,
                ).cpu().numpy()

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


def main():
    TRAIN = True

    if BOUNDED:
        ckpt = "model_weighted_bounded_incremental/theta_{theta_index}/epoch_{step}.pth"
        png_file = "model_weighted_bounded_incremental/theta_{theta_index}/comparison.png"
        losses_file = "model_weighted_bounded_incremental/theta_{theta_index}/losses.png"
    else:
        ckpt = "model_weighted_incremental/theta_{theta_index}/epoch_{step}.pth"
        png_file = "model_weighted_incremental/theta_{theta_index}/comparison.png"
        losses_file = "model_weighted_incremental/theta_{theta_index}/losses.png"

    step = 0
    theta_index = 0

    num_epochs = 200

    T = 1000
    beta_start = 0.00085
    beta_end = 0.012

    num_samples = 200_000
    num_points = 2_000_000
    batch_size = 50_000

    learning_rate = 1e-4
    dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # threshold = 50.5 if BOUNDED else 28
    # theta = find_theta(dist.cpu().numpy(), threshold)
    # theta = torch.tensor(theta, dtype=dtype, device=device)

    # current_step = 0
    model = ScoreNet(input_dim=dim, dtype=dtype, device=device, time_dim=64, num_layers=3)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-6
    )
    losses = []

    if step:
        path = ckpt.format(theta_index=theta_index, step=step)
        print(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        losses = checkpoint["losses"]

    threshold = 50.5 if BOUNDED else 28
    theta = find_theta(dist.cpu().numpy(), threshold)
    theta = torch.tensor(theta, dtype=dtype, device=device)

    last_dist = dist
    if step != 0 and theta_index != 0:
        samples = sample(
            model,
            num_samples,
            betas,
            alphas,
            alpha_bars,
            T,
            device,
            theta=torch.zeros_like(theta),
            dim=dim,
            bounded=BOUNDED,
            batch_size=batch_size*2,
        )
        last_dist = torch.tensor(last_dist, dtype=dtype, device=device)


    if TRAIN:
        while theta_index < THETA_STEPS:
            effective_theta = (
                TARGET_THETA * ((theta_index + 1) / THETA_STEPS) * torch.ones_like(theta)
            )
            theta = (
                (TARGET_THETA / THETA_STEPS) * torch.ones_like(theta)
            )
            print(theta.max().item())
            print((theta**2).sum().sqrt().item())
            weights = torch.exp(torch.sum(theta.view(1, -1) * last_dist, dim=1, keepdim=True))
            dataset = TensorDataset(last_dist, weights)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print(f"Training on theta index {theta_index}/{THETA_STEPS}")
            train(
                model,
                losses,
                dataloader,
                last_dist,
                optimizer,
                scheduler,
                betas,
                alphas,
                alpha_bars,
                num_samples,
                effective_theta,
                T,
                device,
                ckpt=ckpt.format(theta_index=theta_index, step="{step}"),
                png_file=png_file.format(theta_index=theta_index),
                losses_file=losses_file.format(theta_index=theta_index),
                num_epochs=num_epochs,
                start_epoch=step,
                dim=dim,
                bounded=BOUNDED,
            )

            model.eval()
            theta_zero = torch.zeros_like(theta)
            samples = sample(
                model,
                last_dist.shape[0],
                betas,
                alphas,
                alpha_bars,
                T,
                device,
                theta=theta_zero,
                dim=dim,
                bounded=BOUNDED,
                batch_size=batch_size*2,
            )

            # The `theta` here is the tensor for the current iteration: (TARGET_THETA * ((theta_index + 1) / THETA_STEPS) * torch.ones_like(dist))
            # The `dist` here is the original data tensor, which should remain a tensor.
            # We need to convert `theta` and `dist` to numpy specifically for `rejection_sampling`
            current_iter_theta_numpy = theta.cpu().numpy()
            original_dist_numpy = dist.cpu().numpy()

            if BOUNDED:
                inp_dist = rejection_sampling(original_dist_numpy, last_dist.shape[0], theta=current_iter_theta_numpy)
            else:
                inp_dist = create_twisted_multivariate_normal(
                    theta=current_iter_theta_numpy,
                    num_points=last_dist.shape[0],
                    dim=dim,
                    dtype=dtype
                ).cpu().numpy()
            diff, distance = compare_dist(samples, inp_dist)
            for key, value in distance.items():
                print(f"{key}: {value:.4f}", end=" | ")
            print()

            png_file_path = png_file.format(theta_index=theta_index)

            if not os.path.exists(os.path.dirname(png_file_path)):
                os.makedirs(os.path.dirname(png_file_path))

            plot_comparison(
                inp_dist,
                samples,
                "Rejection Sampling Distribution",
                "Diffusion Model Distribution",
                path=png_file_path,
            )

            last_dist = torch.tensor(samples, dtype=dtype, device=device)
            theta_index += 1

            model = ScoreNet(input_dim=dim, dtype=dtype, device=device, time_dim=64, num_layers=3)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_epochs, eta_min=1e-6
            )
            losses = []

    else:
        model.eval()

        theta_zero = torch.zeros((num_samples, dim), dtype=dtype, device=device) # `theta` here is the one from the last iteration of the loop
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
            batch_size=batch_size*2,
        )

        # `theta` from the last loop iteration needs to be numpy for rejection_sampling
        # `dist` (the original data) should be converted to numpy here for rejection_sampling
        final_theta_numpy = ((theta_index + 1)/TARGET_THETA * torch.ones((num_samples, dim), dtype=dtype, device=device)).cpu().numpy()
        original_dist_numpy = dist.cpu().numpy()

        if BOUNDED:
            inp_dist = rejection_sampling(original_dist_numpy, num_samples, theta=final_theta_numpy)
        else:
            inp_dist = create_twisted_multivariate_normal(
                theta=final_theta_numpy,
                num_points=num_samples,
                dim=dim,
                dtype=dtype
            ).cpu().numpy()
        diff, distance = compare_dist(samples, inp_dist)
        for key, value in distance.items():
            print(f"{key}: {value:.4f}", end=" | ")

        print()

        png_file_path = png_file.format(theta_index=theta_index)

        if not os.path.exists(os.path.dirname(png_file_path)):
            os.makedirs(os.path.dirname(png_file_path))

        plot_comparison(
            inp_dist,
            samples,
            "Rejection Sampling Distribution",
            "Diffusion Model Distribution",
            path=png_file_path,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded")
    args = parser.parse_args()

    if args.bounded:
        BOUNDED = True
    else:
        BOUNDED = False

    TARGET_THETA = 0.41613638401031494 if BOUNDED else 0.24622570723295212
    main()
