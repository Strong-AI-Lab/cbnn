
from typing import Optional
import math

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
        self.fc_in_mean.weight.data.normal_(0, 1)
        self.fc_in_log_var.weight.data.normal_(0, 1)

        self.fc_in_bn = torch.nn.BatchNorm1d(hidden_dim)

        self.fc_out_mean = torch.nn.Embedding(hidden_dim, out_dim)
        self.fc_out_log_var = torch.nn.Embedding(hidden_dim, out_dim)
        self.fc_out_mean.weight.data.normal_(0, 1)
        self.fc_out_log_var.weight.data.normal_(0, 1)

        self.fc_hidden = torch.nn.ModuleList()
        self.fc_activations = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.fc_hidden.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.fc_activations.append(torch.nn.SiLU())
            self.fc_norms.append(torch.nn.BatchNorm1d(hidden_dim))

    
    def _sample_weights(self, mean: torch.Tensor, log_var: torch.Tensor, eps: Optional[torch.Tensor] = None):
        std = torch.exp(0.5 * log_var)

        if eps is None:
            eps = torch.randn_like(std)
        else:
            assert eps.shape == std.shape, f"Shape mismatch between eps and std: {eps.shape} != {std.shape}"

        return mean + eps * std
    
    def get_weight_distributions(self):
        return [torch.cat([self.fc_in_mean.weight.view(-1), self.fc_out_mean.weight.view(-1)]), torch.cat([self.fc_in_log_var.weight.view(-1), self.fc_out_log_var.weight.view(-1)])]


    def forward(self, x: torch.Tensor, eps_in: Optional[torch.Tensor] = None, eps_out: Optional[torch.Tensor] = None):
        fc_in_weights = self._sample_weights(self.fc_in_mean.weight, self.fc_in_log_var.weight, eps_in)
        fc_out_weights = self._sample_weights(self.fc_out_mean.weight, self.fc_out_log_var.weight, eps_out)

        x = torch.nn.functional.silu(x @ fc_in_weights)
        x = self.fc_in_bn(x)

        for i in range(self.num_layers):
            x = self.fc_activations[i](self.fc_hidden[i](x)) + x
            x = self.fc_norms[i](x)

        x = x @ fc_out_weights
        return [x, fc_in_weights, fc_out_weights]


class MCQABayesClassifier(BayesianClassifier):
    """
    Bayesian classifier for Multiple Choice Question Answering
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, nb_context: int = 4, nb_choices: int = 4):
        self.single_input_size = in_dim // (nb_context + nb_choices)

        super(MCQABayesClassifier, self).__init__(self.single_input_size, hidden_dim, hidden_dim, num_layers)
        self.nb_context = nb_context
        self.nb_choices = nb_choices

        self.context_merger = torch.nn.Linear(hidden_dim * nb_context, hidden_dim)

    def forward(self, x: torch.Tensor, eps_in: Optional[torch.Tensor] = None, eps_out: Optional[torch.Tensor] = None):
        batch_size  = x.size(0)

        # Compute embeddings of context and choices
        x = x.view(-1, self.single_input_size) # [batch_size, (nb_context + nb_choices) * single_input_size] -> [batch_size *(nb_context + nb_choices), single_input_size]
        x, fc_in_weights, fc_out_weights = super().forward(x, eps_in, eps_out)
        x = x.view(batch_size, self.nb_context + self.nb_choices, self.hidden_dim) # [batch_size, (nb_context + nb_choices) * hidden_dim] -> [batch_size, nb_context + nb_choices, hidden_dim]

        context = x[:, :self.nb_context,:] # [batch_size, nb_context, hidden_dim]
        choices = x[:, self.nb_context:,:] # [batch_size, nb_choices, hidden_dim]

        context = torch.nn.functional.silu(self.context_merger(context.view(batch_size, -1))) #  [batch_size, nb_context, hidden_dim] ->  [batch_size, hidden_dim]

        # Normalise embeddings
        context = context + torch.randn_like(context) * 1e-6 # Add eps*1e-6 to avoid division by zero
        choices = choices + torch.randn_like(choices) * 1e-6
        context = context / context.norm(dim=-1, keepdim=True)
        choices = choices / choices.norm(dim=-1, keepdim=True)

        # Compute scores
        scores = (context.unsqueeze(1) @ choices.permute(0,2,1)).view(batch_size, self.nb_choices) # [batch_size, 1, hidden_dim] x [batch_size, hidden_dim, nb_choices] -> [batch_size, nb_choices]

        return [scores, fc_in_weights, fc_out_weights]