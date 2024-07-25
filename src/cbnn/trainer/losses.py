
from typing import Optional

import torch


def normal_kullback_leibler_divergence(mu : torch.Tensor, log_var : torch.Tensor):
    """
    Compute the KL divergence between a Gaussian distribution with mean mu and log variance log_var and a standard normal distribution
    """
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))


def correlation_coefficient(samples : torch.Tensor):
    """
    Compute the correlation coefficient of a set of samples with shape [B x F]
    """
    # Compute the mean of the samples
    mean = torch.mean(samples, dim=0) # [B x F] -> [F]

    # Compute the centered samples
    centered_samples = samples - mean # [B x F] - [F] -> [B x F]

    # Compute the covariance matrix
    covariance = torch.matmul(centered_samples.t(), centered_samples) / samples.size(0) # [F x B] x [B x F] -> [F x F]

    # Compute the diagonal of the covariance matrix
    diagonal = torch.diag(covariance) # [F x F] -> [F]

    # Compute the standard deviations
    std = torch.sqrt(diagonal) # [F]

    # Compute the correlation coefficient
    return covariance / torch.ger(std, std) # [F x F] / [F x F] -> [F x F]

def smooth_abs(x : torch.Tensor, beta : float = 1e-4):
    """
    Compute the smooth absolute value of x, i.e. sqrt(x^2 + beta^2)
    """
    return torch.sqrt(x**2 + beta**2)

def gaussian_mutual_information(samples_p : torch.Tensor, samples_q : torch.Tensor, max_dim : Optional[int] = None, boost_coefficients : Optional[float] = None, top_k : Optional[int] = None):
    """
    Compute the mutual information between two sets of samples [B x P] and [B x Q]. It is assumed that the samples are drawn from a joint Gaussian multivariate distribution.
    [1] MacKay, David JC. Information theory, inference and learning algorithms. Cambridge university press, 2003. (Chapter 11).
    """
    batch_size = samples_p.size(0)
    assert batch_size == samples_q.size(0), "Batch sizes of samples_p and samples_q must be equal"

    # Reduce dimensions for computational efficiency
    if max_dim is not None:
        samples_p = samples_p[:,torch.randperm(samples_p.size(1))[:max_dim // 2]]
        samples_q = samples_q[:,torch.randperm(samples_q.size(1))[:max_dim // 2]]

    # Compute the correlation coefficient
    correlations = correlation_coefficient(torch.cat([samples_p - samples_p.mean(0), samples_q - samples_q.mean(0)], dim=1)) # [B x (P + Q)] -> [(P + Q) x (P + Q)]

    # Boost coefficients to avoid numerical instability
    if boost_coefficients is not None:
        correlations = correlations.sign() * correlations.abs()**(1/(1+boost_coefficients))

    # Detach inter-distribution correlations (we are not interested in the mutual information between the dimensions of the same distribution)
    mask = torch.ones_like(correlations)
    mask[:samples_p.size(1), :samples_p.size(1)] = 0
    mask[-samples_q.size(1):, -samples_q.size(1):] = 0
    mask_rev = 1 - mask 

    correlations = correlations * mask + (correlations * mask_rev).detach()

    # Keep rows with highest gradient (i.e. highest correlation coefficients)
    if top_k is not None:
        keep_idx = torch.topk((correlations * mask).abs().sum(dim=0), top_k, dim=0).indices
        correlations = correlations[keep_idx][:,keep_idx]

    # Compute the mutual information using the determinant of the correlation matrix, return the smooth absolute value for optimization purposes (we are mot interestedin the sign of the information), similarly use the smooth absolute value of the determinant to avoid numerical instability
    return smooth_abs(-0.5 * smooth_abs(torch.det(correlations)).log()) # [(P + Q) x (P + Q)] -> [1]





    