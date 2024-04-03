
import math
from typing import Optional

import torch


def normal_kullback_leibler_divergence(mu : torch.Tensor, log_var : torch.Tensor):
    """
    Compute the KL divergence between a Gaussian distribution with mean mu and log variance log_var and a standard normal distribution
    """
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))


def normal_continuous_entropy(mu : torch.Tensor, log_var : torch.Tensor, jaynes_correction : bool = True, a : Optional[float] = None, b : Optional[float] = None, samples : Optional[torch.Tensor] = None):
    """
    Compute the continuous entropy of a Gaussian distribution with mean mu and log variance log_var
    If jaynes_correction is True, the Limiting density of discrete points is applied to the entropy to make it invariant to linear transformations of the data
    The invariant measure is the uniform distribution between a and b and the density adjustment is computed using the samples
    """
    # Compute the continuous entropy
    entropy = 0.5 * torch.log(torch.tensor(2 * math.pi * math.e) * log_var.exp())

    if jaynes_correction:
        assert a is not None and b is not None and samples is not None, "If jaynes_correction is True, a, b and samples must be provided"

        # Compute the density adjustment
        density_adjustment = torch.log(torch.abs((b - a))) + torch.mean(samples, dim=1)
        entropy -= density_adjustment

    return entropy.mean()
    

def gaussian_mutual_information(mu_p : torch.Tensor, log_var_p : torch.Tensor, mu_q : torch.Tensor, log_var_q : torch.Tensor, jaynes_correction : bool = True, a : Optional[float] = None, b : Optional[float] = None, samples_p : Optional[torch.Tensor] = None, samples_q : Optional[torch.Tensor] = None):
    """
    Compute the mutual information between two Gaussian distributions with means mu_p and mu_q and log variances log_var_p and log_var_q
    If jaynes_correction is True, the Limiting density of discrete points is applied to the entropy to make it invariant to linear transformations of the data
    The invariant measure is the uniform distribution between a and b and the density adjustment is computed using the samples
    """
    return normal_continuous_entropy(mu_p, log_var_p, jaynes_correction, a, b, samples_p) + normal_continuous_entropy(mu_q, log_var_q, jaynes_correction, a, b, samples_q) - normal_continuous_entropy(torch.cat([mu_p, mu_q], dim=1), torch.cat([log_var_p, log_var_q], dim=1), jaynes_correction, a, b, torch.cat([samples_p, samples_q], dim=1))


    

    # non-trivial for continuous variables, see:
    # [1] InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets [Chen et al., 2016] https://arxiv.org/abs/1606.03657
    # [2] Structured Disentangled Representations [Esmaeili et al., 2018] https://arxiv.org/abs/1804.02086
    # or simply the wikipedia article: https://en.wikipedia.org/wiki/Differential_entropy and https://en.wikipedia.org/wiki/Limiting_density_of_discrete_points
    # basically, differential entropy (for cntinuous variables) is a mistake...
    # but should work for gaussian case, nope not bounded and can be negative...
    # H_N(X) = -∫ p(x) log p(x) / m(x) dx with m(x) the uniform distribution
    # H_N(X) = -KL(p(x) || m(x))
    # I_N(X;Y) = H_N(X) + H_N(Y) - H_N(X,Y) = -KL(p(x) || m(x)) + -KL(p(y) || m(y)) + KL(p(x,y) || m(x,y))



    # discrete case: from implementation of [3] Can Large Language Models Learn Independent Causal Mechanisms? [Gendron et al., 2024] https://github.com/Strong-AI-Lab/modular-lm/blob/main/src/modular_lm/loss/mi.py
    # p_x = logits_p.softmax(dim=-1).mean(dim=0) # [B x C] -> [C], compute P(X)
    # p_y = logits_q.softmax(dim=-1).mean(dim=0) # [B x C] -> [C], compute P(Y)
    # p_x_p_y = torch.einsum("i,j->ij", p_x, p_y) # [C], [C] -> [C x C], compute outer product P(X) ⊗ P(Y)
    # log_p_x_p_y = p_x_p_y.clamp(min=1e-6, max=1-1e-6).log()

    # p_xy = torch.einsum("ij,ik->ijk", logits_p.softmax(dim=-1), logits_q.softmax(dim=-1)) # [B x C], [B x C] -> [B x C x C], compute ∑_S P(X|S) ⊗ P(Y|S) P(S) = ∑_S P(X,Y|S) P(S) = P(X,Y) P(S) for all samples S in batch B as X and Y are conditionally independent given S
    # log_xy =  p_xy.mean(dim=0).clamp(min=1e-6, max=1-1e-6).log() # [B x C x C] -> [C x C], perform summation of above formula

    # mi_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)(
    #     log_xy.view((-1)), # [C x C] -> [CC]
    #     log_p_x_p_y.view((-1)), # [C x C] -> [CC]
    # ).sum()
    # return mi_loss # loss can sometimes be > 1 due to clamp
    