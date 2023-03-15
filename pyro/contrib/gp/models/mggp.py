import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import mggp_conditional
from pyro.distributions.util import eye_like
from pyro.nn.module import PyroParam, pyro_method









class MultigroupGP(GPModel):
    
    def __init__(self, X, y, kernel, groups, noises=None, mean_function=None, group_specific_noise_terms=False, jitter=1e-6):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
        super().__init__(X, y, kernel, mean_function, jitter)

        self.groups = groups
        self.n_groups = len(torch.unique(self.groups)) 
        self.group_specific_noise_terms = group_specific_noise_terms
        self.n_noise_terms = self.n_groups if self.group_specific_noise_terms else 1


        noises = torch.ones(self.n_noise_terms) if noises is None else noises
        assert len(noises)==self.n_noise_terms
        self.noises = PyroParam(noises, constraint=constraints.positive)

    
    @pyro_method
    def model(self):
        self.set_mode("model")

        N = self.X.size(0)
        Kff = self.kernel(self.X, self.groups)

        if self.group_specific_noise_terms:
            for i, group in enumerate(torch.unique(self.groups)):
                Kff.view(-1)[:: N + 1][self.groups==group] += self.jitter + self.noises[i]
        else:
            Kff.view(-1)[:: N + 1] += self.jitter + self.noises[0]

        Lff = torch.linalg.cholesky(Kff)
        zero_loc = self.X.new_zeros(self.X.size(0))
        f_loc = zero_loc + self.mean_function(self.X)

        if self.y is None:
            f_var = Lff.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.MultivariateNormal(f_loc, scale_tril=Lff)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.dim() - 1),
                obs=self.y,
            )

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

    def forward(self, Xnew, groupsXnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        N = self.X.size(0)
        Kff = self.kernel(self.X, self.groups).contiguous()

        if self.group_specific_noise_terms:
            for i, group in enumerate(torch.unique(self.groups)):
                Kff.view(-1)[:: N + 1][self.groups==group] += self.jitter + self.noises[i]

        else:
            Kff.view(-1)[:: N + 1] += self.jitter + self.noises[0]
        Lff = torch.linalg.cholesky(Kff)

        y_residual = self.y - self.mean_function(self.X)
        loc, cov = mggp_conditional(
            Xnew,
            groupsXnew,
            self.X,
            self.groups,
            self.kernel,
            y_residual,
            None,
            Lff,
            full_cov,
            jitter=self.jitter,
        )

        if full_cov and not noiseless:
            M = Xnew.size(0)
            cov = cov.contiguous()
            cov.view(-1, M * M)[:, :: M + 1] += self.noise  # add noise to the diagonal
        if not full_cov and not noiseless:
            cov = cov + self.noise

        return loc + self.mean_function(Xnew), cov

