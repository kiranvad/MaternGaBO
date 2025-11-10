#!/usr/bin/env python3
"""Geometry-aware Bayesian Active Learning with 3D visualisation.

We retain the original hierarchical structure over a product manifold M = C × T,
with C = R^2 (Euclidean coordinates c1, c2) and T a Hilbert-sphere proxy of
functions represented via SRVT-like vectors q. Each evaluation samples a pair
(c, q), but—per the extended demo—we also tap into the reconstructed function
f(c, t) to place five auxiliary points in the 3D space (c1, c2, f). These points
receive +/− labels according to proximity to a hidden centre (c*, q*), letting
us fit a lightweight boundary surrogate for visualisation.

Left panel: sampled 3D points together with a wireframe approximation of the
surrogate decision boundary in (c1, c2, f).
Right panel: function reconstructions visited by the active learner, coloured by
iteration index to inspect convergence.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import ZeroMean
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 kept for side effects

import geoopt
from geoopt.manifolds.base import Manifold


class SRVTHilbertSphere(Manifold):
    """Hilbert sphere of SRVFs with an L2-induced Riemannian structure."""

    name = "HilbertSphereSRVF"
    ndim = 1
    reversible = True

    def __init__(self, L: int, eps: float = 1e-8):
        super().__init__()
        self.L = int(L)
        self.dt = 1.0 / float(L)
        self.eps = eps
        self._north_value = 1.0 / math.sqrt(self.dt)

    def _north(self, like: torch.Tensor) -> torch.Tensor:
        north = torch.zeros_like(like)
        north[..., 0] = self._north_value
        return north

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
        return (u * v).sum(dim=-1, keepdim=keepdim) * self.dt

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True) * self.dt)
        norm_clamped = norm.clamp_min(self.eps)
        projected = x / norm_clamped
        mask = (norm <= self.eps).expand_as(projected)
        if mask.any():
            north = self._north(projected)
            projected = torch.where(mask, north, projected)
        return projected

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inner = self.inner(x, x, u, keepdim=True)
        return u - inner * x

    egrad2rgrad = proju

    def retr(self, x: torch.Tensor, u: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        return self.projx(x + t * u)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
        inner = self.inner(x, u, u, keepdim=keepdim)
        return torch.sqrt(inner.clamp_min(self.eps))

    def expmap(self, x: torch.Tensor, u: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        v = self.proju(x, u)
        nv = self.norm(x, v, keepdim=True).clamp_min(self.eps)
        cos_term = torch.cos(nv * t)
        sin_term = torch.sin(nv * t) / nv
        y = cos_term * x + sin_term * v
        return self.projx(y)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ip = self.inner(x, x, y).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.arccos(ip)
        sin_theta = torch.sin(theta).clamp_min(self.eps)
        diff = y - ip[..., None] * x
        return (theta / sin_theta)[..., None] * diff

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
        ip = self.inner(x, x, y).clamp(-1.0 + self.eps, 1.0 - self.eps)
        return torch.arccos(ip)

    def rand(self, *size, device=None, dtype=None) -> torch.Tensor:
        sample = torch.randn(*size, self.L, device=device, dtype=dtype)
        return self.projx(sample)

    def randvec(self, x: torch.Tensor, *size, device=None, dtype=None) -> torch.Tensor:
        noise = torch.randn(*size, *x.shape, device=device, dtype=dtype)
        tangent = self.proju(x, noise)
        norm = self.norm(x, tangent, keepdim=True).clamp_min(self.eps)
        return tangent / norm

    def zero(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def typicaldist(self) -> float:
        return math.pi


class HilbertSphere:
    """Discretised Hilbert sphere proxy using SRVF vectors and geoopt."""

    def __init__(self, L: int, T_min: float = 20.0, T_max: float = 80.0):
        self.L = int(L)
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.dt = 1.0 / float(L)
        self.manifold = SRVTHilbertSphere(self.L)

    def project(self, q: torch.Tensor) -> torch.Tensor:
        return self.manifold.projx(q)

    def inner(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (q * p).sum(dim=-1) * self.dt

    def dist(self, q: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        ip = torch.clamp(self.inner(q, p), -1.0 + eps, 1.0 - eps)
        return torch.arccos(ip)

    def expmap(self, q: torch.Tensor, v: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        return self.manifold.expmap(q, v)

    def logmap(self, q: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return self.manifold.logmap(q, p)

    def random_q(self, n: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.manifold.rand(n, device=device)

    def reconstruct_f(self, q: torch.Tensor) -> torch.Tensor:
        delta_T = self.T_max - self.T_min
        s = q.sum(dim=-1, keepdim=True) * self.dt
        alpha = torch.where(torch.abs(s) > 1e-8, delta_T / s, torch.zeros_like(s))
        v = alpha * q
        f0 = torch.full_like(v[..., :1], self.T_min)
        f = f0 + torch.cumsum(v * self.dt, dim=-1)
        corr = (self.T_max - f[..., -1:])
        tgrid = torch.linspace(0, 1, self.L, device=q.device).expand_as(f)
        return f + corr * tgrid


class RBF_SphereGeodesic(Kernel):
    """RBF kernel on the Hilbert sphere using geodesic distances."""

    has_lengthscale = True

    def __init__(self, sphere: HilbertSphere, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.sphere = sphere
        self.eps = eps

    def forward(self, x1, x2, diag: bool = False, **params):
        q1 = self._slice_input(x1)
        if diag:
            return torch.ones(q1.shape[:-1], device=q1.device)
        q2 = self._slice_input(x2)
        ip = torch.matmul(q1, q2.transpose(-1, -2)) * self.sphere.dt
        ip = torch.clamp(ip, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.arccos(ip)
        ls = self.lengthscale
        return torch.exp(-0.5 * (theta / ls) ** 2)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel: Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return MultivariateNormal(mean_x, covar_x)


@dataclass
class ProductBall:
    c0: torch.Tensor  # (2,)
    q0: torch.Tensor  # (L,)
    radius: float
    lam: float = 1.0

    _f0_cache: Optional[torch.Tensor] = None

    def _ensure_f0(self, sphere: HilbertSphere, device: torch.device) -> torch.Tensor:
        if self._f0_cache is None:
            with torch.no_grad():
                self._f0_cache = sphere.reconstruct_f(self.q0[None, :].to(device)).squeeze(0)
        return self._f0_cache.to(device)

    def dist(self, c: torch.Tensor, q: torch.Tensor, sphere: HilbertSphere) -> torch.Tensor:
        device = c.device
        f_curve = sphere.reconstruct_f(q)
        centre_curve = self._ensure_f0(sphere, device)
        c0 = self.c0.to(device)
        c_expand = c.unsqueeze(-2).expand(-1, sphere.L, -1)
        centre_c = c0.unsqueeze(0).unsqueeze(0).expand_as(c_expand)
        diff_c = c_expand - centre_c
        diff_f = f_curve.unsqueeze(-1) - centre_curve.unsqueeze(0).unsqueeze(-1)
        dist = torch.sqrt((diff_c ** 2).sum(dim=-1) + (self.lam * diff_f.squeeze(-1)) ** 2)
        return dist.min(dim=-1).values

    def label(self, c: torch.Tensor, q: torch.Tensor, sphere: HilbertSphere) -> torch.Tensor:
        d = self.dist(c, q, sphere)
        return (d <= self.radius).float()

    def dist_3d(
        self,
        c: torch.Tensor,
        f_value: torch.Tensor,
        t_idx: torch.Tensor,
        sphere: HilbertSphere,
        device: torch.device,
    ) -> torch.Tensor:
        c0 = self.c0.to(device)
        centre_curve = self._ensure_f0(sphere, device)
        centre_z = centre_curve[t_idx]
        diff_c = c - c0
        diff_f = f_value - centre_z
        return torch.sqrt((diff_c ** 2).sum(dim=-1) + (self.lam * diff_f) ** 2)

    def label_3d(
        self,
        c: torch.Tensor,
        f_value: torch.Tensor,
        t_idx: torch.Tensor,
        sphere: HilbertSphere,
        device: torch.device,
    ) -> torch.Tensor:
        d = self.dist_3d(c, f_value, t_idx, sphere, device)
        return (d <= self.radius).float()


class HierarchicalAL:
    def __init__(
        self,
        sphere: HilbertSphere,
        lam_T: float = 1.0,
        noise: float = 1e-2,
        device: str = "cpu",
    ):
        self.sphere = sphere
        self.device = torch.device(device)
        self.noise = noise
        self.lam_T = lam_T

        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self.model: Optional[ExactGPModel] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

        self._boundary_x: Optional[torch.Tensor] = None
        self._boundary_y: Optional[torch.Tensor] = None
        self.boundary_model: Optional[ExactGPModel] = None
        self.boundary_likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

        self.curve_history: list[Tuple[int, torch.Tensor]] = []

    # ---------- GP utilities for (c, q) model ----------
    def _build_kernel(self, L: int) -> Kernel:
        base_c = gpytorch.kernels.RBFKernel(ard_num_dims=2, active_dims=slice(0, 2))
        base_c.initialize(lengthscale=0.5)
        base_t = RBF_SphereGeodesic(self.sphere, active_dims=slice(2, 2 + L))
        base_t.initialize(lengthscale=0.6)
        return gpytorch.kernels.ScaleKernel(base_c) * gpytorch.kernels.ScaleKernel(base_t)

    def _ensure_model(self):
        if self.model is None:
            L = self.sphere.L
            kernel = self._build_kernel(L)
            noise = torch.tensor(self.noise, device=self.device)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=noise)
            self.model = ExactGPModel(self._train_x, self._train_y, self.likelihood, kernel).to(self.device)

    def fit_gp(self, iters: int = 200, lr: float = 0.1, verbose: bool = False):
        if self._train_x is None:
            return
        self._ensure_model()
        self.model.set_train_data(self._train_x, self._train_y, strict=False)
        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(iters):
            opt.zero_grad()
            out = self.model(self._train_x)
            loss = -mll(out, self._train_y)
            loss.backward()
            opt.step()
            if verbose and (i % 50 == 0 or i == iters - 1):
                print(f"[GP] iter {i:03d} loss={loss.item():.4f}")

    def predict_logits(self, X: torch.Tensor, need_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model not initialised. Call add_data() and fit_gp() first.")
        self.model.eval()
        self.likelihood.eval()
        X = X.to(self.device)
        context = torch.enable_grad if need_grad else torch.no_grad
        with context():
            posterior = self.model(X)
            mean = posterior.mean
            var = posterior.variance
        if not need_grad:
            mean = mean.detach()
            var = var.detach()
        return mean, var

    # ---------- GP utilities for boundary model ----------
    def _ensure_boundary_model(self):
        if self.boundary_model is None:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=3)
            )
            noise = torch.tensor(5e-2, device=self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=noise)
            self.boundary_likelihood = likelihood
            self.boundary_model = ExactGPModel(self._boundary_x, self._boundary_y, likelihood, kernel).to(self.device)

    def fit_boundary_gp(self, iters: int = 120, lr: float = 0.09):
        if self._boundary_x is None:
            return
        self._ensure_boundary_model()
        self.boundary_model.set_train_data(self._boundary_x, self._boundary_y, strict=False)
        self.boundary_model.train()
        self.boundary_likelihood.train()
        opt = torch.optim.Adam(self.boundary_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.boundary_likelihood, self.boundary_model)
        for _ in range(iters):
            opt.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-3):
                output = self.boundary_model(self._boundary_x)
                loss = -mll(output, self._boundary_y)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict_boundary_logits(self, X: torch.Tensor) -> torch.Tensor:
        if self.boundary_model is None or self.boundary_likelihood is None:
            raise RuntimeError("Boundary model not initialised.")
        self.boundary_model.eval()
        self.boundary_likelihood.eval()
        with gpytorch.settings.cholesky_jitter(1e-3):
            posterior = self.boundary_model(X.to(self.device))
        return posterior.mean

    # ---------- data management ----------
    def add_data(
        self,
        c: torch.Tensor,
        q: torch.Tensor,
        y: torch.Tensor,
        boundary_features: Optional[torch.Tensor] = None,
        boundary_labels: Optional[torch.Tensor] = None,
        iteration: Optional[int] = None,
        curve: Optional[torch.Tensor] = None,
    ):
        x = torch.cat([c, q], dim=-1)
        y_latent = (y * 2 - 1).reshape(-1)
        if self._train_x is None:
            self._train_x = x.to(self.device)
            self._train_y = y_latent.to(self.device)
        else:
            self._train_x = torch.cat([self._train_x, x.to(self.device)], dim=0)
            self._train_y = torch.cat([self._train_y, y_latent.to(self.device)], dim=0)

        if boundary_features is not None and boundary_labels is not None:
            bf = boundary_features.to(self.device)
            by = (boundary_labels.to(self.device) * 2 - 1).reshape(-1)
            if self._boundary_x is None:
                self._boundary_x = bf
                self._boundary_y = by
            else:
                self._boundary_x = torch.cat([self._boundary_x, bf], dim=0)
                self._boundary_y = torch.cat([self._boundary_y, by], dim=0)

        if curve is not None and iteration is not None:
            self.curve_history.append((iteration, curve.detach().cpu()))

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._train_x is None or self._train_y is None:
            raise RuntimeError("Dataset empty. Call add_data() first.")
        c = self._train_x[..., :2]
        q = self._train_x[..., 2:]
        y = (self._train_y + 1) * 0.5
        return c, q, y

    def get_boundary_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._boundary_x is None or self._boundary_y is None:
            return None, None
        xyz = self._boundary_x.detach().cpu().numpy()
        labels = ((self._boundary_y.detach().cpu().numpy() + 1) * 0.5)
        return xyz, labels

    def boundary_range(self) -> Tuple[float, float]:
        if self._boundary_x is None:
            return (self.sphere.T_min, self.sphere.T_max)
        f_vals = self._boundary_x[:, 2]
        return float(f_vals.min().item()), float(f_vals.max().item())

    # ---------- acquisition utilities ----------
    def _entropy_from_logit(self, m: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(m)
        eps = 1e-8
        p = p.clamp(eps, 1 - eps)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    def acquisition(self, c: torch.Tensor, q: torch.Tensor, need_grad: bool = False) -> torch.Tensor:
        X = torch.cat([c, q], dim=-1)
        m, _ = self.predict_logits(X, need_grad=need_grad)
        return self._entropy_from_logit(m)

    @torch.no_grad()
    def outer_over_C(self, C_grid: torch.Tensor, q_samples_per_c: int = 16) -> Tuple[torch.Tensor, float]:
        best_val = -1.0
        best_c = None
        for c in C_grid:
            c_batch = c[None, :].expand(q_samples_per_c, -1)
            q_samp = self.sphere.random_q(q_samples_per_c, device=self.device)
            val = self.acquisition(c_batch, q_samp, need_grad=False).mean().item()
            if val > best_val:
                best_val = val
                best_c = c
        return best_c, best_val

    def inner_over_T(self, c_star: torch.Tensor, steps: int = 100, lr: float = 0.1) -> torch.Tensor:
        manifold = self.sphere.manifold
        q_init = self.sphere.random_q(1, device=self.device).squeeze(0)
        q = geoopt.ManifoldParameter(q_init, manifold=manifold)
        opt = geoopt.optim.RiemannianSGD([q], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            acq = -self.acquisition(c_star[None, :], q[None, :], need_grad=True).mean()
            acq.backward()
            opt.step()
        return q.detach().clone()

    @torch.no_grad()
    def expected_entropy_grid(self, C_grid: torch.Tensor, q_samples_per_c: int = 16) -> torch.Tensor:
        values = []
        for c in C_grid:
            c_batch = c[None, :].expand(q_samples_per_c, -1)
            q_samp = self.sphere.random_q(q_samples_per_c, device=self.device)
            ent = self.acquisition(c_batch, q_samp, need_grad=False).mean()
            values.append(ent)
        return torch.stack(values)

    @torch.no_grad()
    def estimate_boundary_surface(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        f_vals: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.boundary_model is None:
            return None
        xs_t = torch.tensor(xs, dtype=torch.float32, device=self.device)
        ys_t = torch.tensor(ys, dtype=torch.float32, device=self.device)
        f_grid = torch.tensor(f_vals, dtype=torch.float32, device=self.device)
        surface = torch.full((xs_t.numel(), ys_t.numel()), float("nan"), device=self.device)
        for i, c1 in enumerate(xs_t):
            for j, c2 in enumerate(ys_t):
                c = torch.tensor([c1, c2], device=self.device)
                c_batch = c.unsqueeze(0).expand(f_grid.shape[0], -1)
                X = torch.cat([c_batch, f_grid.unsqueeze(-1)], dim=-1)
                logits = self.predict_boundary_logits(X)
                idx = torch.argmin(torch.abs(logits))
                surface[i, j] = f_grid[idx]
        return surface.cpu().numpy()


class ActiveLearningPlotter3D:
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        f_linspace: np.ndarray,
        learner: HierarchicalAL,
        sphere: HilbertSphere,
        max_iters: int,
    ):
        self.xs = xs
        self.ys = ys
        self.f_linspace = f_linspace
        self.learner = learner
        self.sphere = sphere

        self.fig = plt.figure(figsize=(13, 5))
        self.ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.axf = self.fig.add_subplot(1, 2, 2)

        self.ax3d.set_xlabel("c1")
        self.ax3d.set_ylabel("c2")
        self.ax3d.set_zlabel("f(c, t)")
        self.ax3d.set_xlim(xs[0], xs[-1])
        self.ax3d.set_ylim(ys[0], ys[-1])

        self.axf.set_xlabel("t")
        self.axf.set_ylabel("Temperature")
        self.axf.set_ylim(sphere.T_min - 5.0, sphere.T_max + 5.0)

        self.scatter = None
        self.latest = None
        self.boundary_wire = None
        self.iter_colors = plt.cm.plasma(np.linspace(0.15, 0.95, max_iters + 2))
        self.curve_handles = []

        plt.ion()
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.05)

    def update(self, iteration: int, latest_points: Optional[np.ndarray] = None):
        xyz, labels = self.learner.get_boundary_data()
        if xyz is not None:
            if self.scatter is not None:
                self.scatter.remove()
            colors = np.where(labels > 0.5, "#ef4444", "#2563eb")
            self.scatter = self.ax3d.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                c=colors,
                s=40,
                depthshade=False,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.9,
                zorder=4,
            )

        if self.boundary_wire is not None:
            self.boundary_wire.remove()
            self.boundary_wire = None
        surface = self.learner.estimate_boundary_surface(self.xs, self.ys, self.f_linspace)
        if surface is not None:
            X, Y = np.meshgrid(self.xs, self.ys, indexing="xy")
            self.boundary_wire = self.ax3d.plot_wireframe(
                X,
                Y,
                surface.T,
                color="#111827",
                linewidth=1.0,
                alpha=0.85,
                rstride=2,
                cstride=2,
                zorder=3,
            )

        if self.latest is not None:
            self.latest.remove()
            self.latest = None
        if latest_points is not None:
            self.latest = self.ax3d.scatter(
                latest_points[:, 0], latest_points[:, 1], latest_points[:, 2],
                c="#facc15",
                marker="X",
                s=90,
                depthshade=False,
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )

        self.ax3d.set_title(f"3D samples & boundary (iter {iteration})")
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def add_curve(self, t: np.ndarray, values: np.ndarray, iteration: int):
        color = self.iter_colors[min(iteration, len(self.iter_colors) - 1)]
        (handle,) = self.axf.plot(t, values, color=color, linewidth=2.0, alpha=0.95, label=f"iter {iteration:02d}")
        self.curve_handles.append(handle)
        self.axf.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, ncol=2)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)


def sample_function_points(
    c: torch.Tensor,
    q: torch.Tensor,
    sphere: HilbertSphere,
    truth: ProductBall,
    n_points: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f_curve = sphere.reconstruct_f(q.unsqueeze(0)).squeeze(0)
    idx = torch.randperm(sphere.L, device=q.device)[:n_points]
    f_vals = f_curve[idx]
    c_rep = c.unsqueeze(0).expand(n_points, -1)
    labels = truth.label_3d(c_rep, f_vals, idx, sphere, q.device)
    features = torch.stack([c_rep[:, 0], c_rep[:, 1], f_vals], dim=-1)
    return features, labels, idx, f_curve


def demo(
    seed: int = 0,
    L: int = 64,
    n_init: int = 12,
    iters: int = 50,
    device: str = "cpu",
    q_samples_per_c: int = 32,
    n_t_points: int = 5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device_obj = torch.device(device)
    sphere = HilbertSphere(L=L, T_min=20.0, T_max=80.0)

    c0 = torch.tensor([0.3, -0.2], dtype=torch.float32, device=device_obj)
    q0 = sphere.random_q(1, device=device_obj).squeeze(0)
    truth = ProductBall(c0=c0, q0=q0, radius=1.5, lam=1.0)

    learner = HierarchicalAL(sphere=sphere, lam_T=1.0, noise=1e-2, device=device)

    # Provide one seed at the hidden centre to anchor the classifiers
    centre_features, centre_labels, _, centre_curve = sample_function_points(c0, q0, sphere, truth, n_points=n_t_points)
    learner.add_data(
        c0.unsqueeze(0),
        q0.unsqueeze(0),
        torch.ones(1, device=device_obj),
        boundary_features=centre_features,
        boundary_labels=centre_labels,
        iteration=0,
        curve=centre_curve,
    )

    # Initial random design
    C_init = torch.empty(n_init, 2, device=device_obj).uniform_(-1.0, 1.0)
    Q_init = sphere.random_q(n_init, device=device_obj)
    y_init = truth.label(C_init, Q_init, sphere)

    for idx_sample in range(n_init):
        c_i = C_init[idx_sample]
        q_i = Q_init[idx_sample]
        y_i = y_init[idx_sample:idx_sample + 1]
        feat_i, lab_i, _, curve_i = sample_function_points(c_i, q_i, sphere, truth, n_points=n_t_points)
        learner.add_data(c_i.unsqueeze(0), q_i.unsqueeze(0), y_i.unsqueeze(0), boundary_features=feat_i, boundary_labels=lab_i)

    learner.fit_gp(iters=200, lr=0.1, verbose=False)
    learner.fit_boundary_gp(iters=180, lr=0.08)

    xs = np.linspace(-1.0, 1.0, 25)
    ys = np.linspace(-1.0, 1.0, 25)
    f_min, f_max = learner.boundary_range()
    f_linspace = np.linspace(f_min - 1.0, f_max + 1.0, 60)
    t_dense = torch.linspace(0.0, 1.0, sphere.L, device=device_obj)

    plotter = ActiveLearningPlotter3D(xs, ys, f_linspace, learner, sphere, max_iters=iters)
    plotter.update(iteration=0)

    C_grid = torch.tensor(
        np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2),
        dtype=torch.float32,
        device=device_obj,
    )

    for t in range(iters):
        c_star, _ = learner.outer_over_C(C_grid, q_samples_per_c=q_samples_per_c)
        c_star = c_star.to(device_obj)
        q_star = learner.inner_over_T(c_star, steps=120, lr=0.15)

        y_star = truth.label(c_star[None, :], q_star[None, :], sphere)
        feat_star, lab_star, t_idx_star, curve_star = sample_function_points(c_star, q_star, sphere, truth, n_points=n_t_points)

        learner.add_data(
            c_star[None, :],
            q_star[None, :],
            y_star,
            boundary_features=feat_star,
            boundary_labels=lab_star,
            iteration=t + 1,
            curve=curve_star,
        )
        learner.fit_gp(iters=120, lr=0.07, verbose=False)
        learner.fit_boundary_gp(iters=140, lr=0.07)

        plotter.add_curve(t_dense.cpu().numpy(), curve_star.detach().cpu().numpy(), iteration=t + 1)
        plotter.update(iteration=t + 1, latest_points=feat_star.detach().cpu().numpy())

        with torch.no_grad():
            dists = truth.dist(c_star[None, :], q_star[None, :], sphere).item()
            dists_aux = truth.dist_3d(
                feat_star[:, :2],
                feat_star[:, 2],
                t_idx_star,
                sphere,
                device_obj,
            ).mean().item()
        print(
            f"[iter {t+1:02d}] y={int(y_star.item())} dist_cq={dists:.3f} avg_dist_3d={dists_aux:.3f}"
        )

    return learner, sphere, truth


if __name__ == "__main__":
    demo()
