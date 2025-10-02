"""
# Geodesics between Warping Functions using the SRSF Representation

## Background

A **warping function** is a diffeomorphism  
\[
\gamma : [0,1] \to [0,1], \quad \gamma(0)=0, \ \gamma(1)=1, \ \dot\gamma(t) > 0.
\]

Its **Square-Root Slope Function (SRSF)** representation is defined as:
\[
q(t) = \sqrt{\dot\gamma(t)}.
\]

This representation maps the space of warpings to the **unit sphere** in the Hilbert space \(L^2([0,1])\), because
\[
\|q\|^2 = \int_0^1 q(t)^2 dt = \int_0^1 \dot\gamma(t)\,dt = \gamma(1) - \gamma(0) = 1.
\]

Thus, computing geodesics between warping functions reduces to computing great-circle geodesics on a Hilbert sphere.

---

## Inner Product and Angle

Given two SRSFs \(q_a\) and \(q_b\), their inner product is
\[
\langle q_a, q_b \rangle = \int_0^1 q_a(t)\, q_b(t)\, dt.
\]

The geodesic angle between them is
\[
\theta = \arccos\big(\langle q_a, q_b \rangle\big).
\]

---

## Geodesic Interpolation on the Sphere

For interpolation parameter \(s \in [0,1]\), the geodesic path is given by the great-circle formula:
\[
q_s(t) = \frac{\sin\big((1-s)\theta\big)}{\sin\theta}\, q_a(t) \;+\; 
          \frac{\sin(s\theta)}{\sin\theta}\, q_b(t).
\]

Special cases:
- If \(\theta \approx 0\): \(q_s \approx (1-s)q_a + s q_b\) (linear interpolation + renormalization).
- If \(\theta = \pi\): choose a perpendicular direction \(v\) to define the geodesic:
  \[
  q_s(t) = \cos(\pi s)\, q_a(t) + \sin(\pi s)\, v(t).
  \]

---

## Recovering Warping Functions

Each intermediate SRSF \(q_s\) can be mapped back to a warping function via
\[
\gamma_s(x) = \int_0^x q_s(u)^2 \, du.
\]

This ensures \(\gamma_s(0)=0\) and \(\gamma_s(1)=1\), preserving the diffeomorphism property.

---

## Summary

1. Convert warping functions \(\gamma_a, \gamma_b\) to SRSFs \(q_a, q_b\).
2. Normalize \(q_a, q_b\) to lie on the unit \(L^2\)-sphere.
3. Compute the geodesic angle \(\theta = \arccos(\langle q_a, q_b\rangle)\).
4. Interpolate intermediate SRSFs using the great-circle formula.
5. Map back to warping functions via cumulative integration:
   \[
   \gamma_s(x) = \int_0^x q_s(u)^2 \, du.
   \]

This procedure yields the **geodesic path of warping functions** under the elastic metric induced by the SRSF representation.

"""
import torch

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

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = "cpu"  # or "cuda" if you want GPU
    N = 501
    t = torch.linspace(0, 1, N, device=device, dtype=torch.float64)

    # Example gammas
    gamma_a = t.clone()
    gamma_b = t**2

    dt = t[1] - t[0]
    def derivative(x, dt):
        dx = torch.zeros_like(x)
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        return dx

    gd_a = derivative(gamma_a, dt)
    gd_b = derivative(gamma_b, dt)
    q_a = torch.sqrt(torch.clamp(gd_a, min=0.0))
    q_b = torch.sqrt(torch.clamp(gd_b, min=0.0))

    s_vals, Qs, Gammas = geodesic_between_srsfs(q_a, q_b, t, n_steps=11, device=device)

    plt.figure(figsize=(6,5))
    for i in [0, 2, 5, 8, 10]:
        plt.plot(t.cpu().numpy(), Gammas[i].cpu().numpy(), label=f"s={s_vals[i].item():.2f}")
    plt.plot(t.cpu().numpy(), gamma_a.cpu().numpy(), "k--", label="gamma_a")
    plt.plot(t.cpu().numpy(), gamma_b.cpu().numpy(), "k-.", label="gamma_b")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("gamma(t)")
    plt.title("Geodesic between warping functions (PyTorch)")
    plt.show()
