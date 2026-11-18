import torch
import math
import torch.nn as nn
from einops import rearrange,einsum,reduce
from torch import Tensor
from jaxtyping import Float,Bool,Int
from cs336_basics.transformer_block_module import softmax
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"],dim=-1
) -> Float[Tensor, ""]:
    # use targets as slice
    # here is a problem, although we have already considered about numeric stability in softmax
    # how ever, that is to avoid overflow
    # but here we need to take log to them
    # if one is underflow to 0 at that time, then it will be -inf after taking log
    # boom!
    # def logsumexp(x, dim):
    # m = x.amax(dim, keepdim=True)              # 每行最大值
    # return (x - m).exp().sum(dim, keepdim=True).log() + m
    batch_size, vocab_size = inputs.shape
    lse = torch.logsumexp(inputs, dim=dim, keepdim=False)# torch.logsumexp avoid this problem
    gt = inputs[torch.arange(batch_size),targets] # avoid one log element underflow
    return torch.mean(-gt + lse)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def  __init__(self,params,lr,betas,eps,weight_decay):
        defaults = {"alpha":lr,"beta1":betas[0],"beta2":betas[1],"weight_decay":weight_decay,'eps':eps}
        super().__init__(params,defaults)

    def step(self,closure:Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group['alpha']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                t = state.get("t",0) + 1
                m = state.get("m",torch.zeros_like(p))
                v = state.get('v',torch.zeros_like(p))
                alpha_t = alpha * math.sqrt(1-beta2**t) / (1 - beta1**t)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                p.data.add_(
                    -alpha * weight_decay * p.data
                    - alpha_t * m / (torch.sqrt(v) + eps)
                )# in-place operation
                state['m'] = m
                state['v'] = v
                state['t'] = t
        return loss

def lr_cosine_schedule(t:int,alpha_max:float,alpha_min:float,tw:int,tc:int):
    if t < tw:
        return t * alpha_max / tw
    elif t > tc:
        return alpha_min
    else:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - tw) / (tc - tw))) * (alpha_max - alpha_min)

def gradient_clipping(parameters,M,eps=1e-6):
    total_norm = torch.sqrt(        
        sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None)
    )
    if total_norm >= M:        
        scale = M / (total_norm + eps)        
        for p in parameters:            
            if p.grad is not None:                
                p.grad.mul_(scale)