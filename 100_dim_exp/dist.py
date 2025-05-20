import math
import os
import torch
import numpy as np
from typing import Dict, Sequence
from scipy import special as sc
import concurrent.futures as cf


def create_multivariate_normal(
    num_points: int = 1000,
    dim: int = 9,
    random_state: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    if dim == 1:
        z = np.random.normal(size=num_points)
        z = torch.tensor(z, dtype=dtype).unsqueeze(1)
        return z

    rho_path = f"npy/rhos_{dim}.npy"
    file_dir = os.path.dirname(rho_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    num_rhos = dim * (dim - 1) // 2
    if os.path.exists(rho_path):
        rhos = np.load(rho_path)
        if rhos.shape[0] != num_rhos:
            raise ValueError(
                f"Stored rhos_{dim}.npy has wrong length "
                f"{rhos.shape[0]} (expected {num_rhos}). "
                "Delete the file to regenerate."
            )
    else:
        # Build a random SPD correlation matrix then extract its off-diagonals
        A = np.random.randn(dim, dim)
        cov = A @ A.T  # symmetric positive-definite
        d = np.sqrt(np.diag(cov))
        R = cov / np.outer(d, d)  # convert to correlation matrix

        # Flatten the strict upper triangle into a vector
        rhos = R[np.triu_indices(dim, k=1)]
        np.save(rho_path, rhos)

    R = np.eye(dim)
    tri_rows, tri_cols = np.triu_indices(dim, k=1)
    R[tri_rows, tri_cols] = rhos
    R[tri_cols, tri_rows] = rhos  # symmetry

    # ---------- sample ----------
    z_np = np.random.multivariate_normal(
        mean=np.zeros(dim),
        cov=R,
        size=num_points,
    )

    return torch.tensor(z_np, dtype=dtype)


def beta_icdf(u, a, b, *, device=None, dtype=None):
    if hasattr(torch.special, "betaincinv"):  # GPU → 0.3 s
        return torch.special.betaincinv(a, b, u)

    u_np, a_np, b_np = (t.detach().cpu().numpy() for t in (u, a, b))
    out_np = np.empty_like(u_np)
    CHUNK = 5_000
    sls = [slice(i, i + CHUNK) for i in range(0, u_np.size, CHUNK)]
    with cf.ThreadPoolExecutor() as pool:
        pool.map(
            lambda s: out_np.__setitem__(s, sc.betaincinv(a_np[s], b_np[s], u_np[s])),
            sls,
        )
    return torch.as_tensor(out_np, device=device or u.device, dtype=dtype or u.dtype)


def create_multivariate_beta_using_copula(
    num_points: int = 100_000,
    dim: int = 9,
    *,
    alpha: float | Sequence[float] | None = None,
    beta: float | Sequence[float] | None = None,
    random_state: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    # 1) multivariate normal with desired correlations
    x = create_multivariate_normal(
        num_points=num_points,
        dim=dim,
        random_state=random_state,
        dtype=dtype,
    )  # (n, d) torch tensor

    # 2) Φ : N(0,1) → U(0,1)
    u = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))  # (n, d) in (0,1)

    # 3) inverse-Beta CDF (icdf) per dimension
    #    – allow scalar or per-dimension α, β
    if np.isscalar(alpha):
        alpha = [float(alpha)] * dim
    if np.isscalar(beta):
        beta = [float(beta)] * dim

    if alpha is None:
        alpha_path = f"npy/alphas_{dim}.npy"
        file_dir = os.path.dirname(alpha_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if os.path.exists(alpha_path):
            alphas = np.load(alpha_path)
            if alphas.shape[0] != dim:
                raise ValueError(
                    f"Stored alphas_{dim}.npy has wrong length "
                    f"{alphas.shape[0]} (expected {dim}). "
                    "Delete the file to regenerate."
                )
            alpha = alphas.tolist()
        else:
            choices = np.arange(1, 5.5, 0.5)
            alpha = np.random.choice(choices, size=dim, replace=True).tolist()
            np.save(alpha_path, alpha)

    if beta is None:
        beta_path = f"npy/betas_{dim}.npy"
        file_dir = os.path.dirname(beta_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if os.path.exists(beta_path):
            betas = np.load(beta_path)
            if betas.shape[0] != dim:
                raise ValueError(
                    f"Stored betas_{dim}.npy has wrong length "
                    f"{betas.shape[0]} (expected {dim}). "
                    "Delete the file to regenerate."
                )
            beta = betas.tolist()
        else:
            choices = np.arange(1, 5.5, 0.5)
            beta = np.random.choice(choices, size=dim, replace=True).tolist()
            np.save(beta_path, beta)

    if len(alpha) != dim or len(beta) != dim:
        raise ValueError("alpha and beta must be scalars or sequences of length `dim`")

    a_t = torch.tensor(alpha, dtype=dtype, device=u.device).unsqueeze(0)  # (1,d)
    b_t = torch.tensor(beta, dtype=dtype, device=u.device).unsqueeze(0)  # (1,d)

    # Vectorised icdf: build independent Beta distrib. for each column
    y = beta_icdf(u, a_t.expand_as(u), b_t.expand_as(u))

    return y


def _rff(x: torch.Tensor, gamma: float, n_feat: int, *, laplace=False):
    """
    Random Fourier Features  for either Gaussian (rbf) or Laplacian kernels
    k(x,y)=exp(-γ‖x-y‖₂²)             or  exp(-γ‖x-y‖₂).
    """
    d = x.size(1)
    device, dtype = x.device, x.dtype
    if not laplace:  # Gaussian ⇒ w ~ N(0, 2γ I)
        w = torch.randn(d, n_feat, device=device, dtype=dtype) * math.sqrt(2 * gamma)
    else:  # Laplace ⇒ w ~ Cauchy(0, γ)
        w = (
            torch.distributions.Cauchy(0.0, math.sqrt(2 * gamma))
            .sample((d, n_feat))
            .to(device=device, dtype=dtype)
        )
    b = 2 * math.pi * torch.rand(n_feat, device=device, dtype=dtype)
    z = math.sqrt(2.0 / n_feat) * torch.cos(x @ w + b)  # (n, n_feat)
    return z


def compare_dist(
    dist1,
    dist2,
    gammas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    n_feat: int = 512,
    device: str | torch.device = "cpu",
    dtype=torch.float32,
) -> Dict[str, float]:
    # --- harmonise inputs -------------------------------------------------- #
    x, y = (
        torch.as_tensor(dist1, device=device, dtype=dtype),
        torch.as_tensor(dist2, device=device, dtype=dtype),
    )

    res = {}

    mu_x, mu_y = x.mean(dim=0), y.mean(dim=0)
    res["mean"] = torch.dot(mu_x - mu_y, mu_x - mu_y).item()

    var_x, var_y = x.var(dim=0), y.var(dim=0)
    res["variance"] = torch.dot(var_x - var_y, var_x - var_y).item()

    def _mean_feature(x: torch.Tensor, feat_fn, **kw):
        return feat_fn(x, **kw).mean(0)  # (n_feat,)

    # ---- 2) shift-invariant kernels via RFF ------------------------------- #
    for g in gammas:
        # Gaussian / RBF
        phi_x = _mean_feature(x, _rff, gamma=g, n_feat=n_feat, laplace=False)
        phi_y = _mean_feature(y, _rff, gamma=g, n_feat=n_feat, laplace=False)
        res[f"rbf_γ={g}"] = torch.sum((phi_x - phi_y) ** 2).item()

        # Laplacian
        psi_x = _mean_feature(x, _rff, gamma=g, n_feat=n_feat, laplace=True)
        psi_y = _mean_feature(y, _rff, gamma=g, n_feat=n_feat, laplace=True)
        res[f"laplacian_γ={g}"] = torch.sum((psi_x - psi_y) ** 2).item()

    distance_vector = np.array(list(res.values()))
    l2_distance = np.linalg.norm(distance_vector)
    l1_distance = np.sum(np.abs(distance_vector))

    mu_diff = (mu_x - mu_y).cpu().numpy()
    var_diff = (var_x - var_y).cpu().numpy()

    mu_distance = np.sum(np.abs(mu_diff))
    var_distance = np.sum(np.abs(var_diff))

    distance = {
        "L1": l2_distance,
        "L2": l1_distance,
        "MU": mu_distance,
        "VAR": var_distance,
    }

    return res, distance


def rejection_sampling(
    dist,
    num_samples: int,
    theta: np.ndarray,
):
    dim = dist.shape[1]
    assert dim == theta.shape[0], "Dimension mismatch between dist and theta"

    # Sample from the exponentially tilted distribution using rejection sampling (p_theta(x) \propto exp(theta^T x))
    weights = np.exp(theta @ dist.T)
    weights /= np.sum(weights)

    idx = np.random.choice(dist.shape[0], size=num_samples, replace=True, p=weights)
    return dist[idx]
