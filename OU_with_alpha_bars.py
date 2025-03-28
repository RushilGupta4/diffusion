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
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=5):
        super(ScoreNet, self).__init__()
        # The input layer takes the 2D point concatenated with a scalar time feature.
        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # x: (B, 2) noisy 2D points; t: (B,) normalized timestep in [0,1]
        t = t.view(-1, 1)  # reshape time to (B, 1)
        x = torch.cat([x, t], dim=1)
        return self.net(x)


def get_diffusion_hyperparams(T, beta_start, beta_end, device):
    """Creates the beta schedule and corresponding cumulative products."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def create_circle_dataset(num_points=1000, radius=5.0):
    """Generates a circle dataset with given number of points and radius."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    data = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)
    return torch.tensor(data, dtype=torch.float32)


def train(model, dataloader, optimizer, alpha_bars, T, device, num_epochs=10, ckpt="model_{step}.pth"):
    """Training loop for the diffusion model with a single tqdm progress bar per epoch."""
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_steps = T * num_batches
        
        # Create a single tqdm progress bar for the entire epoch
        with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for _ in range(T):
                for (x0,) in dataloader:
                    x0 = x0.to(device)
                    B = x0.shape[0]

                    # Sample random timesteps for each point and normalize to [0, 1]
                    t = torch.randint(0, T, (B,), device=device)
                    t_norm = t.float() / T
                    t_norm = t_norm.unsqueeze(1)

                    # OU process forward (with OU rate λ; here we choose λ=1.0)
                    # and train the denoiser on θ = 0 (i.e. no additional drift during training)
                    noise = torch.randn_like(x0)
                    x_t = x0 * torch.exp(-t_norm) + torch.sqrt(1 - torch.exp(-2 * t_norm)) * noise

                    # Normalize timestep to [0, 1]
                    noise_pred = model(x_t, t_norm)
                    loss = F.mse_loss(noise_pred, noise)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * B
                    pbar.update(1)
        

        samples = sample(model, num_samples=500, alpha_bars=alpha_bars, T=T, device=device, theta=np.array([0, 0]))
        mean_x = samples[:, 0].mean()
        mean_y = samples[:, 1].mean()
        eval_score = mean_x ** 2 + mean_y ** 2
        if eval_score < best_loss:
            best_loss = eval_score
            path = ckpt.format(step=epoch)
            torch.save(model.state_dict(), path)
        
        visualize_results(samples)

        epoch_loss /= (dataset_size * T)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Mean: ({mean_x:.4f}, {mean_y:.4f}), Eval Score: {eval_score:.6f}, Best: {best_loss:.6f}\n\n")


@torch.no_grad()
def sample(model, num_samples, alpha_bars, T, device, theta):
    model.eval()
    x = torch.randn(num_samples, 2, device=device)  # x0^R ~ N(0, I)
    dt = 1.0 / T  # Simple discretization step
    
    # Ensure theta is a tensor on the correct device and with appropriate shape
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, device=device, dtype=torch.float32)
    theta = theta.view(1, -1)  # Make it broadcastable, e.g. (1,2)
    
    for i in reversed(range(T)):
        # Current time normalized to [0, 1]
        t = torch.full((num_samples,), i / T, device=device)
        
        # Compute the two components of the modified score:
        # 1. The explicit theta term: theta * exp(t)
        term1 = theta * torch.exp(t).unsqueeze(1)  # shape: (num_samples, 2)
        
        # 2. The learned score evaluated at a shifted position: x + (t - exp(-t))*theta
        shifted_input = x + (torch.exp(t).unsqueeze(1) - torch.exp(-t).unsqueeze(1)) * theta
        noise_pred = model(shifted_input, t)
        score = - noise_pred / torch.sqrt(1 - alpha_bars[i])
        
        # Combine to form the modified score function:
        score = term1 + score
        
        # Euler-Maruyama update for the reverse SDE:
        noise = torch.randn_like(x) if i > 0 else 0
        x = x + (x + score) * dt + torch.sqrt(torch.tensor(2 * dt, device=device)) * noise
        
    return x


def visualize_results(samples, save_path="diffusion_samples.png"):
    """Plots the original circle data and generated samples."""
    # Convert tensors to numpy arrays for plotting if needed.
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.6)
    plt.title("Generated 2D Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig(save_path)
    plt.close()


def main():
    TRAIN = True
    step = 6
    ckpt = "model_{step}.pth"

    # Hyperparameters
    T = 400  # total number of diffusion steps
    beta_start = 0.0001
    beta_end = 0.02

    num_samples = 1000
    num_points = 512
    batch_size = 256

    num_epochs = 20
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, alpha_bars = get_diffusion_hyperparams(T, beta_start, beta_end, device)
    
    # Create the dataset and dataloader
    data_circle = create_circle_dataset(num_points=num_points, radius=5.0)

    if TRAIN:
        # Create the dataloader
        dataset = TensorDataset(data_circle)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model and optimizer
        model = ScoreNet(input_dim=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train(model, dataloader, optimizer, alpha_bars, T, device, num_epochs=num_epochs, ckpt=ckpt)
    
    else:
        # Load the trained model
        path = ckpt.format(step=step)
        model = ScoreNet(input_dim=2).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

    # Generate samples using the reverse process
    theta = np.array([1, 1]) * 1
    samples = sample(model, num_samples=num_samples, alpha_bars=alpha_bars, T=T, device=device, theta=theta)

    # Print the averages of the mean x and y values of the actual data by weighing each point (x, y) by exp(theta^T * (x, y))
    data_circle = data_circle.cpu().numpy() # Dimension: (num_points, 2)
    theta_np = np.array(theta)                  # shape: (2,)
    weights = np.exp(np.dot(data_circle, theta_np))  # shape: (num_points,)
    weighted_avg = np.sum(data_circle * weights[:, None], axis=0) / np.sum(weights)
    print(f"Weighted Average x: {weighted_avg[0]:.4f}, Weighted Average y: {weighted_avg[1]:.4f}")
   
    # Print the averages of the mean x and y values of the generated samples
    print(f"\nAverage x: {samples[:, 0].mean():.4f}, Average y: {samples[:, 1].mean():.4f}")

    # Visualize the original and generated data
    visualize_results(samples)

if __name__ == '__main__':
    main()