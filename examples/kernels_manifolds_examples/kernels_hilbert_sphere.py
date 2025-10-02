import numpy as np
import random
import torch
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from BoManifolds.Riemannian_utils.hilbert_sphere_utils_torch import hilbert_sphere_distance_torch
from BoManifolds.kernel_utils.kernels_hilbert_sphere import HilbertSphereApproximatedGaussianKernel
from BoManifolds.Riemannian_utils.hilbert_sphere_utils_torch import geodesic_between_srsfs

from funcshape.functions import Function, SRSF

device = 'cpu'

if __name__ == "__main__":
    seed = 1234
    # Set numpy and pytorch seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def warp_start(t, device="cpu", dtype=torch.float64):
        a = torch.empty(1, device=device, dtype=dtype).uniform_(-3, 3).item()
        if abs(a) < 1e-12:   # handle a ≈ 0
            gam = t.clone()
        else:
            gam = (torch.exp(a * t) - 1.0) / (torch.exp(torch.tensor(a, dtype=dtype, device=device)) - 1.0)
        return gam


    def warp_end(x, a, dtype=torch.float64):
        a = torch.as_tensor(a, dtype=dtype, device=x.device)
        term1 = 0.5 * torch.log(a * x + 1.0) / torch.log(a + 1.0)
        term2 = 0.25 * (1.0 + torch.tanh(a * (x - 0.5)) / torch.tanh(a + 1.0))
        return term1 + term2

    n_domain = 100
    t = torch.linspace(0, 1, n_domain, device=device, dtype=torch.float64)

    # sample warping functions and 
    # create SRSF functions based on those
    start = Function(t, warp_start(t).reshape(-1,1))
    q_start = SRSF(start).qx
    end = Function(t, warp_end(t, 15.0).reshape(-1,1))
    q_end = SRSF(end).qx

    # Now the SRSF functions lie on HilberSphere, 
    # we can inherit its differential geoemtry
    s_vals, data_srsf, data = geodesic_between_srsfs(
        q_start.squeeze(), 
        q_end.squeeze(), 
        t,
        n_steps=21
        )

    x1 = data.to(device)
    x2 = q_end.T.to(device)
    print(x1.shape, x2.shape)

    lengthscale = 0.5
    # RBF kernel
    rbf_kernel = HilbertSphereApproximatedGaussianKernel()
    rbf_kernel.beta = 1./lengthscale**2
    rbf_kernel.to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Euclidean Matérn kernel
    euclidean_matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
    euclidean_matern_kernel.lengthscale = lengthscale
    euclidean_matern_kernel.to(device)
    K_euclidean_matern = euclidean_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Distance between x1 and x2
    distance = hilbert_sphere_distance_torch(x1, x2)
    distance_np = distance.cpu().detach().numpy()

    # Plot kernel value in function of the distance
    fig, axs = plt.subplots(1,3, figsize=(4*3, 4))
    fig.subplots_adjust(wspace=0.5)
    # --- Left axis: RBF kernel ---
    axs[0].plot(distance_np, 
                K_rbf, 
                label=r"$\mathbb{S}_{\infty}$", 
                color="tab:blue"
            )
    axs[0].set_xlabel(r'$d_{\mathbb{S}_{\infty}}(x1,x2)$')
    axs[0].set_ylabel(r'$k_{\mathbb{S}_{\infty}}(x1,x2)$', color="tab:blue")
    axs[0].tick_params(axis='y', labelcolor="tab:blue")

    # --- Right axis: Euclidean kernel ---
    ax2 = axs[0].twinx()
    line2, = ax2.plot(distance_np, 
                      K_euclidean_matern, 
                      label=r"$\mathbb{R}^{d}$", 
                      color="tab:red"
                      )
    ax2.set_ylabel(r'$k_{\mathbb{R}^{d}}(x1,x2)$', color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Optional: combined legend (both curves)
    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2, loc="best")

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0, vmax=data.shape[0])
    # plot SRSFs geodesics
    axs[2].plot(t, start.fx, color="k", alpha=0.25)
    axs[2].plot(t, end.fx, color="k", alpha=0.25, ls='--')
    for i in range(data.shape[0]):
        axs[2].plot(t, data[i,:],  color=cmap(norm(i) ))
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("f(t)")

    # Plot warping function geodesics
    axs[1].plot(t, q_start, color="k", alpha=0.25)
    axs[1].plot(t, q_end, color="k", alpha=0.25, ls='--')
    for i in range(data.shape[0]):
        axs[1].plot(t, data_srsf[i,:],color=cmap(norm(i)))
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("q(t)")
    plt.show()
