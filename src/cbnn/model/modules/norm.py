
import torch


class BatchNorm2dNoTrack(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        kwargs.pop('track_running_stats', None)
        super(BatchNorm2dNoTrack, self).__init__(*args,  **kwargs, track_running_stats=False)

        self.running_mean = torch.nn.Parameter(torch.zeros(self.num_features), requires_grad=False) # Parameters are unused but required for the state_dict
        self.running_var = torch.nn.Parameter(torch.ones(self.num_features), requires_grad=False)
        self.num_batches_tracked = torch.nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key not in state_dict:
            state_dict[num_batches_tracked_key] = self.num_batches_tracked

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
class BatchNorm1dNoTrack(torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        kwargs.pop('track_running_stats', None)
        super(BatchNorm1dNoTrack, self).__init__(*args,  **kwargs, track_running_stats=False)

        self.running_mean = torch.nn.Parameter(torch.zeros(self.num_features), requires_grad=False) # Parameters are unused but required for the state_dict
        self.running_var = torch.nn.Parameter(torch.ones(self.num_features), requires_grad=False)
        self.num_batches_tracked = torch.nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key not in state_dict:
            state_dict[num_batches_tracked_key] = self.num_batches_tracked

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )



class FilterResponseNormalization(torch.nn.Module):
    """
    Filter Response Normalization
    Follows principles from [1]
    [1] Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks, Singh, S., & Krishnan. 2020
    """

    def __init__(self, num_features: int, eps: float = 1e-6, tau: float = 1e-6):
        super(FilterResponseNormalization, self).__init__()

        self.eps = eps
        self.tau = tau

        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor):
        x_shape = x.shape # [B, C, H, W]
        x_width = x_shape[2]
        x_height = x_shape[3]

        x = x.view(x_shape[0], x_shape[1], x_width * x_height)

        # Filter Response Normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        gamma = self.gamma.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        x = x * gamma + beta

        x = x.view(x_shape)

        # Threshold Linear Unit
        tau = torch.ones_like(x) * self.tau
        x = torch.stack([x, tau], dim=-1).max(dim=-1)[0]

        return x