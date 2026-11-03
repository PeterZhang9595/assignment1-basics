import torch
import math
import torch.nn as nn
from einops import rearrange,einsum
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, device=None, dtype=None):
        super().__init__()
        # shape of W is [out_features, in_features]
        self.mean = 0.0
        self.std = math.sqrt(2 / (in_features + out_features))

        if weight is not None:
            # 用外部权重：只校验形状，不重置
            if weight.shape != (out_features, in_features):
                raise ValueError(
                    f"weight shape {tuple(weight.shape)} != "
                    f"{(out_features, in_features)}"
                )
            w = weight.detach().clone().to(device=device, dtype=dtype)
        else:
            # 随机初始化
            w = torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
            nn.init.trunc_normal_(
                w, mean=self.mean, std=self.std,
                a=-3 * self.std, b=3 * self.std,
            )

        self.weight = nn.Parameter(w)
        self.shape = self.weight.shape

    @classmethod
    def from_pretrained(cls, in_features, out_features, weight, **kwargs):
        # 不需要额外初始化，全部交给 __init__
        return cls(in_features, out_features, weight=weight, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weight=None, device=None, dtype=None):
        super().__init__()
        # shape: [num_embeddings, embedding_dim]
        self.mean = 0.0
        self.std = 1

        if weight is not None:
            if weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    f"weight shape {tuple(weight.shape)} != "
                    f"{(num_embeddings, embedding_dim)}"
                )
            w = weight.detach().clone().to(device=device, dtype=dtype)
        else:
            w = torch.empty(
                size=(num_embeddings, embedding_dim), device=device, dtype=dtype
            )
            nn.init.trunc_normal_(
                w, mean=self.mean, std=self.std,
                a=-3 * self.std, b=3 * self.std,
            )

        self.weight = nn.Parameter(w)
        self.shape = self.weight.shape

    @classmethod
    def from_pretrained(cls, num_embeddings, embedding_dim, weight, **kwargs):
        return cls(num_embeddings, embedding_dim, weight=weight, **kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: LongTensor, 任意形状 [...]
        # 返回: [..., embedding_dim]
        return self.weight[token_ids]