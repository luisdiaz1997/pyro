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


class VariationalSparseMGGGP(GPModel):
    
    def __init__(self, 
                 X, 
                 y, 
                 kernel, 
                 groups, 
                 Xu,
                 Xu_groups, 
                 likelihood,
                 mean_function=None,
                 latent_shape=None,
                 num_data=None,
                 whiten=False,
                 jitter=1e-6):
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

        self.likelihood = likelihood
        self.Xu = Parameter(Xu)
        self.Xu_groups = Xu_groups

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        M = self.Xu.size(0)
        u_loc = self.Xu.new_zeros(self.latent_shape + (M,))
        self.u_loc = Parameter(u_loc)

        identity = eye_like(self.Xu, M)
        u_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.u_scale_tril = PyroParam(u_scale_tril, constraints.lower_cholesky)

        self.num_data = num_data if num_data is not None else self.X.size(0)
        self.whiten = whiten
        self._sample_latent = True
    
    @pyro_method
    def model(self):
        self.set_mode("model")

        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu, self.Xu_groups).contiguous()
        Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
        Luu = torch.linalg.cholesky(Kuu)

        zero_loc = self.Xu.new_zeros(self.u_loc.shape)
        if self.whiten:
            identity = eye_like(self.Xu, M)
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(zero_loc, scale_tril=identity).to_event(
                    zero_loc.dim() - 1
                ),
            )
        else:
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(zero_loc, scale_tril=Luu).to_event(
                    zero_loc.dim() - 1
                ),
            )

        f_loc, f_var = mggp_conditional(
            self.X,
            self.groups,
            self.Xu,
            self.Xu_groups,
            self.kernel,
            self.u_loc,
            self.u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )

        f_loc = f_loc + self.mean_function(self.X)
        if self.y is None:
            return f_loc, f_var
        else:
            # we would like to load likelihood's parameters outside poutine.scale context
            self.likelihood._load_pyro_samples()
            with poutine.scale(scale=self.num_data / self.X.size(0)):
                return self.likelihood(f_loc, f_var, self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

        pyro.sample(
            self._pyro_get_fullname("u"),
            dist.MultivariateNormal(self.u_loc, scale_tril=self.u_scale_tril).to_event(
                self.u_loc.dim() - 1
            ),
        )

    def forward(self, Xnew, Xnew_groups, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).

        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned from a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        loc, cov = mggp_conditional(
            Xnew,
            Xnew_groups,
            self.Xu,
            self.Xu_groups,
            self.kernel,
            self.u_loc,
            self.u_scale_tril,
            full_cov=full_cov,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        return loc + self.mean_function(Xnew), cov

