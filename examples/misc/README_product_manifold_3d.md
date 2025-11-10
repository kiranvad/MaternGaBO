# Geometry-aware Bayesian Active Learning on a Product Manifold

This note summarises the mathematical construction implemented in
`examples/misc/02_product_manifold_al.py`. The example couples a geometry-aware
Bayesian active learner on the product manifold

\[
\mathcal{M} = \mathcal{C} \times \mathcal{T}, \qquad \mathcal{C} = \mathbf{R}^2,
\]

with an auxiliary surrogate that visualises the induced decision boundary in a
3D feature space \((c_1,c_2,f)\). The demo targets a synthetic binary
classification task defined by a "product ball" around a hidden centre
\((c_0,q_0)\) with radius \(r\).

## Geometry of the product manifold

- **Euclidean part (\(\mathcal{C}\))**: each point is \(c=(c_1,c_2) \in
  \mathbf{R}^2\). Distances are standard Euclidean: \(d_{\mathcal{C}}(c,c') =
  \|c-c'\|_2\).

- **Functional part (\(\mathcal{T}\))**: functions are represented through an
  SRVT-like discretisation \(q \in \mathbf{R}^L\) restricted to the unit sphere
  \(S^{L-1}\). This proxy captures the square-root slope function of a
  temperature curve. The Hilbert sphere metric is the spherical geodesic
  distance
  \[
  d_{\mathcal{T}}(q,q') = \arccos \bigl( \langle q, q'\rangle \bigr).
  \]

- **Product distance**: for a scaling parameter \(\lambda\), the combined metric
  on \(\mathcal{M}\) is
  \[
  d_{\mathcal{M}} \bigl( (c,q), (c',q') \bigr) =
  \sqrt{ d_{\mathcal{C}}(c,c')^2 + \lambda^2 d_{\mathcal{T}}(q,q')^2 }.
  \]

- **Product ball**: the ground-truth label is +1 when
  \(d_{\mathcal{M}}((c,q),(c_0,q_0)) \le r\) and -1 otherwise. The quantity is
  evaluated both for the latent SRVT representation and for the reconstructed
  temperature values (see below).

## Gaussian-process surrogate on \(\mathcal{M}\)

Let \(x = (c,q) \in \mathbf{R}^{2+L}\) be the concatenated representation.
The kernel is a product of an RBF on the Euclidean coordinates and a spherical
RBF on the SRVT vectors:

\[
K_{\mathcal{M}}(x,x') = K_{\mathcal{C}}(c,c') \cdot K_{\mathcal{T}}(q,q'),
\]

with

- \(K_{\mathcal{C}}(c,c') = \exp(-\tfrac{1}{2}\|c-c'\|^2 / \ell_C^2)\);
- \(K_{\mathcal{T}}(q,q') = \exp(-\tfrac{1}{2} d_{\mathcal{T}}(q,q')^2 / \ell_T^2)\).

The learner wraps this kernel in a standard exact GP regression model with a
Gaussian likelihood. Labels \(y\in\{0,1\}\) are mapped to latent logits
\(z = 2y-1\) so that the posterior mean approximates the logit of the class
probability.

### Posterior predictions

For a batch \(X\) the model yields posterior mean \(m(X)\) and variance
\(\sigma^2(X)\). The binary decision surrogate uses a sigmoid link
\(p(X)=\sigma(m(X))\).

## Hierarchical acquisition rule

Each active-learning step maximises the predictive entropy

\[
H(X) = -p(X) \log p(X) - (1-p(X)) \log (1-p(X))
\]

of the latent Bernoulli variable.

1. **Outer loop over \(\mathcal{C}\)**: enumerate a fixed Cartesian grid
   \(\{c_j\}_j\). For each candidate we Monte-Carlo average the entropy over
   fresh SRVT samples \(q\sim \text{Unif}(S^{L-1})\). The maximiser is the
   next spatial query \(c^*\).

2. **Inner loop over \(\mathcal{T}\)**: with \(c^*\) fixed, we optimise \(q\)
   on the sphere using Riemannian SGD provided by `geoopt`. Gradients propagate
   through the GP posterior thanks to the `need_grad=True` switch in
   `predict_logits`.

The result is a pair \((c^*, q^*)\) queried against the ground truth. The
label is appended to the GP training set, and the model is refit by maximising
the exact marginal likelihood.

## 3D auxiliary surrogate

To visualise the decision boundary, each query also samples function values.
Given the SRVT vector \(q\), we reconstruct the temperature profile
\(f_q(t)\) via numerical integration enforcing end-point constraints. A random
subset of indices \(t_i\) (default: 5 per query) yields points in
\(\mathbf{R}^3\):

\[
x_i^{(3D)} = (c_1, c_2, f_q(t_i)).
\]

These points inherit +/− labels using the same product-ball distance but where
the functional component is replaced by \(|f_q(t_i) - f_{q_0}(t_i)|\). A second
GP with an RBF kernel in \(\mathbf{R}^3\) is trained on these samples to
provide a smooth estimate of the boundary: for a grid of \((c_1,c_2)\) points
and candidate function values \(f\), the surrogate predicts the logit, and the
isocontour at 0 approximates the separating surface.

## Demo configuration

The example in `demo()` is configured as follows:

- Hilbert-sphere resolution: \(L=64\), temperature range \([20,80]\).
- Hidden centre: draw \(q_0\) at random, set \(c_0=(0.3,-0.2)\).
- Product-ball parameters: radius \(r=1.5\), scaling \(\lambda=1.0\).
- Initial design: 12 random \((c,q)\) pairs + one seeded example at the true
  centre to ensure the GP has positive evidence.
- At each iteration, the learner samples 5 time indices to create the 3D
  boundary points.
- Outer grid: 25×25 lattice covering \([-1,1]^2\).
- Inner optimiser: Riemannian SGD for 120 steps with step size 0.15.
- GP hyperparameters are refit after each query (Adam with learning rates
  0.07–0.1 for 120–200 iterations).
- Visualisation: the left panel shows the accumulated 3D samples coloured by
  label, together with the surrogate boundary wireframe. The right panel plots
  reconstructed curves coloured by iteration index to monitor convergence.

The console output reports the observed class label, the distance in the
product manifold, and the average distance of the auxiliary 3D samples to the
hidden centre. These diagnostics help to assess whether the acquisition rule is
homing in on the positive region.

## Running the demo

Activate the project environment and execute:

```bash
MPLCONFIGDIR=/tmp/mpl python examples/misc/02_product_manifold_al.py
```

Matplotlib opens an interactive window with the two panels described above. Use
the figure toolbar to rotate the 3D plot for a better view of the boundary.

