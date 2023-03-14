# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam


def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


def _embed_distance_matrix(distance_matrix):

    # Code adapted from https://github.com/andrewcharlesjones/multi-group-GP
    N = len(distance_matrix)
    D2 = distance_matrix**2
    C = torch.eye(N) - 1/N *torch.ones(size=(N, N))
    B = -0.5*C @ D2 @ C
    L, Q = torch.linalg.eigh(B)
    embedding = Q @ torch.diag(_torch_sqrt(L, 1e-6))
    return embedding


def _squared_dist(X, Z):
    
    X2 = (X**2).sum(1, keepdim=True)
    Z2 = (Z**2).sum(1, keepdim=True)
    XZ = X.matmul(Z.t())
    r2 = X2 - 2 * XZ + Z2.t()
    return r2.clamp(min=0)


class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    By default, the parameter ``lengthscale`` has size 1. To use the isotropic version
    (different lengthscale for each dimension), make sure that ``lengthscale`` has size
    equal to ``input_dim``.

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)
        

    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale

        return _squared_dist(scaled_X, scaled_Z)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        return self.variance.expand(X.size(0))


class MultiGroupRBF(Isotropy):


    def __init__(self, input_dim, variance=None, lengthscale=None, group_diff_param=None, group_distances=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)


        self.group_diff_param = torch.tensor(1.0) if group_diff_param is None else group_diff_param
        self.group_distances = group_distances
        self.embedding = None



    def forward(self, X, groupsX, Z=None, groupsZ=None, diag=False):

        if diag:
            return self._diag(X)

        if Z is None:
            Z  = X
            groupsZ = groupsX

        if self.group_distances is None:
            n_groups = len(torch.unique(groupsX))
            self.group_distances = torch.ones(n_groups) - torch.eye(n_groups)

        #TODO, Check if groupsX and groupsY are of integer class

        if self.embedding is None:
            self.embedding = _embed_distance_matrix(self.group_distances)

        group_embeddingsX = self.embedding[groupsX]
        group_embeddingsZ = self.embedding[groupsZ]

        group_r2 = _squared_dist(group_embeddingsX, group_embeddingsZ)
        r2 = self._square_scaled_dist(X, Z)
        assert r2.shape == group_r2.shape

        
        scale = 1 / (self.group_diff_param * group_r2 + 1)**(0.5*self.input_dim)


        return self.variance * torch.exp(-0.5 * r2/ (self.group_diff_param * group_r2 + 1)) * scale

    

class RBF(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \sigma^2\exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r2 = self._square_scaled_dist(X, Z)
        return self.variance * torch.exp(-0.5 * r2)


class RationalQuadratic(Isotropy):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \sigma^2 \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """

    def __init__(
        self,
        input_dim,
        variance=None,
        lengthscale=None,
        scale_mixture=None,
        active_dims=None,
    ):
        super().__init__(input_dim, variance, lengthscale, active_dims)

        if scale_mixture is None:
            scale_mixture = torch.tensor(1.0)
        self.scale_mixture = PyroParam(scale_mixture, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r2 = self._square_scaled_dist(X, Z)
        return self.variance * (1 + (0.5 / self.scale_mixture) * r2).pow(
            -self.scale_mixture
        )


class Exponential(Isotropy):
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        return self.variance * torch.exp(-r)


class Matern32(Isotropy):
    r"""
    Implementation of Matern32 kernel:

        :math:`k(x, z) = \sigma^2\left(1 + \sqrt{3} \times \frac{|x-z|}{l}\right)
        \exp\left(-\sqrt{3} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        sqrt3_r = 3**0.5 * r
        return self.variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)


class Matern52(Isotropy):
    r"""
    Implementation of Matern52 kernel:

        :math:`k(x,z)=\sigma^2\left(1+\sqrt{5}\times\frac{|x-z|}{l}+\frac{5}{3}\times
        \frac{|x-z|^2}{l^2}\right)\exp\left(-\sqrt{5} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r2 = self._square_scaled_dist(X, Z)
        r = _torch_sqrt(r2)
        sqrt5_r = 5**0.5 * r
        return self.variance * (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)
