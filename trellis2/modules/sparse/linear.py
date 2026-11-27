import torch
import torch.nn as nn
from . import VarLenTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        from .inference_chunk import pointwise, active_policy
        policy = active_policy()
        if policy is not None and policy.preserve_linear_shapes:
            if torch.is_grad_enabled():
                raise RuntimeError('Chunked pointwise operations cannot run with gradients enabled')
            return input.replace(super().forward(input.feats))
        return input.replace(pointwise(super().forward, input.feats,
                                       channels=max(self.in_features, self.out_features)))
