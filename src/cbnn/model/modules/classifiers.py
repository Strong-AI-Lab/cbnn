
from typing import Optional

import torch
    

class MLPClassifier(torch.nn.Module):
    """
    Dense classifier with residual skip connections
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 3):
        super(MLPClassifier, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_in = torch.nn.Linear(in_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, out_dim)

        self.fc_hidden = torch.nn.ModuleList()
        for i in range(num_layers):
            self.fc_hidden.append(torch.nn.Linear(hidden_dim, hidden_dim))


    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.leaky_relu(self.fc_in(x))

        for i in range(self.num_layers):
            x = torch.nn.functional.leaky_relu(self.fc_hidden[i](x)) + x

        x = self.fc_out(x)
        return x
    



class BayesianClassifier(torch.nn.Module):
    """
    Fully Differentiable Bayesian classifier
    Follows principles from [1,2]
    [1] Weight Uncertainty in Neural Networks, Blundell et al. 2015
    [2] Do Bayesian Neural Networks Need To Be Fully Stochastic?, Sharma et al. 2023
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 3):
        super(BayesianClassifier, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_in_mean = torch.nn.Embedding(in_dim, hidden_dim)
        self.fc_in_log_var = torch.nn.Embedding(in_dim, hidden_dim)

        self.fc_out_mean = torch.nn.Embedding(hidden_dim, out_dim)
        self.fc_out_log_var = torch.nn.Embedding(hidden_dim, out_dim)

        self.fc_hidden = torch.nn.ModuleList()
        for i in range(num_layers):
            self.fc_hidden.append(torch.nn.Linear(hidden_dim, hidden_dim))

    
    def _sample_weights(self, mean: torch.Tensor, log_var: torch.Tensor, eps: Optional[torch.Tensor] = None):
        std = torch.exp(0.5 * log_var)

        if eps is None:
            eps = torch.randn_like(std)
        else:
            assert eps.shape == std.shape, "Shape mismatch between eps and std"

        return mean + eps * std


    def forward(self, x: torch.Tensor, eps_in: Optional[torch.Tensor] = None, eps_out: Optional[torch.Tensor] = None):
        fc_in_weights = self._sample_weights(self.fc_in_mean.weight, self.fc_in_log_var.weight, eps_in)
        fc_out_weights = self._sample_weights(self.fc_out_mean.weight, self.fc_out_log_var.weight, eps_out)

        x = torch.nn.functional.leaky_relu(x @ fc_in_weights)

        for i in range(self.num_layers):
            x = torch.nn.functional.leaky_relu(self.fc_hidden[i](x)) + x

        x = x @ fc_out_weights
        return [x, fc_in_weights, fc_out_weights]