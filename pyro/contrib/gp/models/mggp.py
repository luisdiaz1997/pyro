import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.distributions.util import eye_like
from pyro.nn.module import PyroParam, pyro_method









class MultigroupGP(GPModel):
    
    def __init__(self, X, y, kernel, groups, group_distances=None, noise=None, mean_function=None, group_specific_noise_terms=False, jitter=1e-6):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
        super().__init__(X, y, kernel, mean_function, jitter)

        self.groups = groups
        self.n_groups = len(torch.unique(groups))
        self.group_distances = group_distances 
        self.group_specific_noise_terms = group_specific_noise_terms
        self.n_noise_terms = self.n_groups if self.group_specific_noise_terms else 1


        noise = self.X.new_tensor(1.0) if noise is None else noise
        self.noises = []

      
        for _ in range(self.n_noise_terms):
            self.noises.append(PyroParam(noise.detach().clone(), constraints.positive))

    
    @pyro_method
    def model(self):
        self.set_mode("model")

        N_total = self.X.size(0)