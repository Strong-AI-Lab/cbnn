
import math
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


def gaussian_mutual_information(samples_p : torch.Tensor, samples_q : torch.Tensor):
    """
    Compute the mutual information between two sets of samples [B x P] and [B x Q]. It is assumed that the samples are drawn from a joint Gaussian multivariate distribution.
    """
    batch_size = samples_p.size(0)
    assert batch_size == samples_q.size(0), "Batch sizes of samples_p and samples_q must be equal"

    # Compute the correlation coefficient
    correlations = correlation_coefficient(torch.cat([samples_p, samples_q], dim=1)) # [B x (P + Q)] -> [(P + Q) x (P + Q)]

    # Mask inter-distribution correlations (we are not interested in the mutual information between the dimensions of the same distribution)
    mask = torch.ones_like(correlations)
    mask[:samples_p.size(1), :samples_p.size(1)] = 0
    mask[-samples_q.size(1):, -samples_q.size(1):] = 0
    correlations = correlations * mask
    correlations = correlations + torch.eye(correlations.size(0)).to(correlations.device)

    # Compute the mutual information
    return -0.5 * torch.logdet(correlations) # [(P + Q) x (P + Q)] -> [1]




    