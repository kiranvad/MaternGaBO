import os
import torch
import gpytorch
from gpytorch.constraints import Positive
import math 

from BoManifolds.Riemannian_utils.hilbert_sphere_utils_torch import hilbert_sphere_distance_torch

dirname = os.path.dirname(os.path.realpath(__file__))
device = "cpu"
torch.set_default_dtype(torch.float32)

class HilbertSphereApproximatedGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.
    This covariance matrix is an approximation of the SphereRiemannianGaussianKernel, where the Euclidean distance is
    replaced by the geodesic distance in a Euclidean-like RBF kernel.

    Attributes
    ----------
    self.dim, dimension of the sphere S^d on which the data handled by the kernel are living
    self.beta_min, minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, beta_prior=None, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param beta: value of the inverse square lengthscale parameter beta.
                        default 1.0.
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super().__init__(has_lengthscale=False, **kwargs)

        self.dim = math.inf

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        self.register_constraint("raw_beta", Positive())

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self._set_beta(value)

    def _set_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = hilbert_sphere_distance_torch(x1, x2, diag=diag)
        distance2 = torch.mul(distance, distance)
        # Kernel
        exp_component = torch.exp(- distance2.mul(self.beta.double()))
        return exp_component