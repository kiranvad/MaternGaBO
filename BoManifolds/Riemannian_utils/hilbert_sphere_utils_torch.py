import torch

def srsf_distance(q_a, q_b, t, eps=1e-10):
    """
    Compute geodesic (great-circle) distance between two SRSFs on the Hilbert sphere.

    Parameters
    ----------
    q_a, q_b : torch.Tensor
        SRSFs of shape (N,) defined on same grid t.
    t : torch.Tensor
        Grid points in [0,1], shape (N,).
    eps : float
        Small tolerance to avoid numerical issues.

    Returns
    -------
    dist : torch.Tensor (scalar)
        Great-circle distance between q_a and q_b.
    """
    # ensure tensors
    q_a = q_a.to(dtype=torch.float64)
    q_b = q_b.to(dtype=torch.float64)
    t   = t.to(dtype=torch.float64)

    # inner product with trapezoidal rule
    ip = torch.trapz(q_a * q_b, t)

    # normalize (in case q_a, q_b not exactly unit norm)
    norm_a = torch.sqrt(torch.trapz(q_a * q_a, t))
    norm_b = torch.sqrt(torch.trapz(q_b * q_b, t))
    ip = ip / (norm_a * norm_b + eps)

    # clamp to [-1,1] for numerical stability
    ip = torch.clamp(ip, -1.0, 1.0)

    # distance = angle
    return torch.arccos(ip)


def hilbert_sphere_distance_torch(x1, x2, diag: bool = False):
    """
    Compute pairwise warping SRSF distances between batches of functions.

    Args:
        x1 (torch.Tensor): (..., N, D) functions
        x2 (torch.Tensor): (..., M, D) functions
        diagonal (bool): If True, only compute diagonal (N == M) distances,
                         return (..., N). Otherwise, return full matrix (..., N, M).

    Returns:
        torch.Tensor: Distance matrix (..., N, M) or (..., N) if diagonal=True.
    """
    device = x1.device
    D = x1.shape[-1]
    domain = torch.linspace(0.0, 1.0, D, device=device)

    *batch_shape, N, _ = x1.shape
    M = x2.shape[-2]

    if diag:
        assert N == M, "Diagonal mode requires x1 and x2 to have same length (N == M)."
        out = []
        for x1_i, x2_i in zip(x1.reshape(-1, N, D), x2.reshape(-1, M, D)):
            d_list = []
            for f1, f2 in zip(x1_i, x2_i):
                d_list.append(torch.tensor(srsf_distance(f1, f2, domain), dtype=x1.dtype, device=x1.device))
            out.append(torch.stack(d_list, dim=0))
        out = torch.stack(out, dim=0)
        return out.reshape(*batch_shape, N)

    else:
        out = []
        for x1_i, x2_i in zip(x1.reshape(-1, N, D), x2.reshape(-1, M, D)):
            d_matrix = []
            for f1 in x1_i:
                row = []
                for f2 in x2_i:
                    row.append(torch.tensor(srsf_distance(f1, f2, domain), dtype=x1.dtype, device=x1.device))
                d_matrix.append(torch.stack(row, dim=0))
            out.append(torch.stack(d_matrix, dim=0))
        out = torch.stack(out, dim=0)
        return out.reshape(*batch_shape, N, M)

def inner_product(q1, q2, t):
    """
    L2 inner product between q1 and q2 on grid t using trapezoidal rule.
    """
    return torch.trapz(q1 * q2, t)

def srsf_to_gamma(q, t):
    """
    Convert SRSF q to warping function gamma defined on same grid t.
    gamma(0)=0, gamma(x) = ∫ q(s)^2 ds
    """
    integrand = q**2
    gamma = torch.zeros_like(t)
    gamma[1:] = torch.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * (t[1:] - t[:-1]),
        dim=0
    )
    return gamma

def _find_perpendicular_unit_vector(q):
    """
    Deterministically find a unit vector orthogonal to q.
    """
    i = torch.argmin(torch.abs(q))
    e = torch.zeros_like(q)
    e[i] = 1.0
    proj = torch.dot(e, q) * q
    v = e - proj
    norm_v = torch.norm(v)
    if norm_v < 1e-12:
        v = torch.randn_like(q)
        v -= torch.dot(v, q) * q
        norm_v = torch.norm(v)
        if norm_v < 1e-12:
            raise RuntimeError("Unable to find a perpendicular vector.")
    return v / norm_v

def geodesic_between_srsfs(q_a, q_b, t, n_steps=21, eps=1e-8, device="cpu"):
    """
    Compute geodesic (list of SRSFs and corresponding gamma functions) 
    between q_a and q_b on grid t using PyTorch.
    """
    q_a = torch.as_tensor(q_a, dtype=torch.float64, device=device)
    q_b = torch.as_tensor(q_b, dtype=torch.float64, device=device)
    t   = torch.as_tensor(t,   dtype=torch.float64, device=device)

    # normalize
    ip_aa = inner_product(q_a, q_a, t)
    ip_bb = inner_product(q_b, q_b, t)
    q_a = q_a / torch.sqrt(ip_aa)
    q_b = q_b / torch.sqrt(ip_bb)

    dot = inner_product(q_a, q_b, t)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)

    s_vals = torch.linspace(0., 1., n_steps, device=device, dtype=torch.float64)
    Qs = []
    Gammas = []

    if torch.isclose(theta, torch.tensor(0.0, device=device, dtype=torch.float64), atol=1e-10):
        for s in s_vals:
            Qs.append(q_a)
            Gammas.append(srsf_to_gamma(q_a, t))
        return s_vals, torch.stack(Qs), torch.stack(Gammas)

    if torch.isclose(dot, torch.tensor(-1.0, device=device, dtype=torch.float64), atol=1e-8):
        v = _find_perpendicular_unit_vector(q_a)
        for s in s_vals:
            q_s = torch.cos(torch.pi * s) * q_a + torch.sin(torch.pi * s) * v
            q_s = q_s / torch.sqrt(inner_product(q_s, q_s, t))
            Qs.append(q_s)
            Gammas.append(srsf_to_gamma(q_s, t))
        return s_vals, torch.stack(Qs), torch.stack(Gammas)

    sin_theta = torch.sin(theta)
    if sin_theta < eps:
        for s in s_vals:
            q_s = (1 - s) * q_a + s * q_b
            q_s = q_s / torch.sqrt(inner_product(q_s, q_s, t))
            Qs.append(q_s)
            Gammas.append(srsf_to_gamma(q_s, t))
        return s_vals, torch.stack(Qs), torch.stack(Gammas)

    for s in s_vals:
        w1 = torch.sin((1.0 - s) * theta) / sin_theta
        w2 = torch.sin(s * theta) / sin_theta
        q_s = w1 * q_a + w2 * q_b
        q_s = q_s / torch.sqrt(inner_product(q_s, q_s, t))
        Qs.append(q_s)
        Gammas.append(srsf_to_gamma(q_s, t))

    return s_vals, torch.stack(Qs), torch.stack(Gammas)
