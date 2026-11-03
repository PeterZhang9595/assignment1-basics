import torch
import math
import torch.nn as nn
from torch import Tensor
from einops import rearrange,einsum,reduce
from jaxtyping import Float,Bool
from cs336_basics.linear import Linear

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,weight=None,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        #self.eps.to(device)
        
        if weight is not None:
            if weight.shape != (d_model,):
                raise ValueError(
                    f"weight shape {tuple(weight.shape)} != "
                    f"{(d_model,)}"
                )
            w = weight.detach().clone().to(device=device, dtype=dtype)
        else:
            w = torch.ones(size=(d_model,),device=device,dtype=dtype)
        self.weight = nn.Parameter(w)
        self.shape = self.weight.shape

    @classmethod
    def from_pretrained(cls,d_model,weight,**kwargs):
        return cls(d_model,weight=weight,**kwargs)


    
    def forward(self,x:Float[torch.Tensor,"batch_size sequence_length d_model"]):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = torch.sqrt(reduce(x.pow(2),"... d -> ... 1","mean") + self.eps)
        rms_norm_x = x / rms_x * self.weight 
        return rms_norm_x.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x *  torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff=None,w1_weight=None,w2_weight=None,w3_weight=None,device=None,dtype=None):
        super().__init__()
        # specification: user can choose either load all weights or load nothing
        if d_ff is None:
            self.d_ff = 64 * round(d_model * 8 / 3 / 64)
        else:
            self.d_ff = d_ff
        self.d_model = d_model

        self.linear1 = None
        self.linear2 = None
        self.linear3 = None

        if any(w is None for w in (w1_weight, w2_weight, w3_weight)):
            self.linear1 = Linear(self.d_model,self.d_ff,weight=None,device=device,dtype=dtype)
            self.linear2 = Linear(self.d_ff,self.d_model,weight=None,device=device,dtype=dtype)
            self.linear3 = Linear(self.d_model,self.d_ff,weight=None,device=device,dtype=dtype)
        else:
            self.linear1 = Linear.from_pretrained(self.d_model,self.d_ff,weight=w1_weight,device=device,dtype=dtype)
            self.linear2 = Linear.from_pretrained(self.d_ff,self.d_model,weight=w2_weight,device=device,dtype=dtype)
            self.linear3 = Linear.from_pretrained(self.d_model,self.d_ff,weight=w3_weight,device=device,dtype=dtype)

    @classmethod
    def from_pretrained(cls,d_model,d_ff,w1_weight,w2_weight,w3_weight,**kwargs):
        return cls(d_model,d_ff,w1_weight,w2_weight,w3_weight,**kwargs)

    def forward(self,x:torch.Tensor):
        return self.linear2.forward((silu(self.linear1.forward(x)) * (self.linear3.forward(x))))



class RoPE(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        k = torch.arange(0,d_k,2,dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (k / d_k))
        i = torch.arange(0,max_seq_len,dtype=torch.float32)
        frequences = torch.outer(i,inv_freq)
        
        cos_cache = torch.cos(frequences).repeat_interleave(2,dim=-1)
        sin_cache = torch.sin(frequences).repeat_interleave(2,dim=-1)

        self.register_buffer("cos_cache",cos_cache,persistent=False)
        self.register_buffer("sin_cache",sin_cache,persistent=False)

    
    def rotate_half(self,x:torch.Tensor):
        x1,x2 = x.chunk(2,dim=-1)
        return torch.cat((-x2,x1),dim=-1)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        cos = self.cos_cache[token_positions].to(x.dtype)
        sin = self.sin_cache[token_positions].to(x.dtype)

        x1 = x[...,0::2]
        x2 = x[...,1::2]
        rotated_x = torch.stack((-x2,x1),dim=-1).flatten(-2)

        return x * cos + rotated_x * sin

def softmax(x:torch.Tensor,i):
    """
    apply softmax to the i-th dimension of the input tensor x.
    the output tensor have the same shape as the input tensor
    but its i-th dimension will now have a normalized probability distribution.
    """
    max_element = torch.amax(x,dim=i,keepdim=True)
    x_minus_max_element = x - max_element
    exp_x = torch.exp(x_minus_max_element)
    exp_sum = torch.sum(exp_x,dim=i,keepdim=True)

    return exp_x / exp_sum

def calculate_activation_score(Q,K,mask):
    d_k = Q.shape[-1]
    temp = (Q @ K.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is None:
        return softmax(x=temp,i=-1)
    else:
        #keys = K.shape[-2]
        temp = temp.masked_fill(~mask,float('-inf'))
        return softmax(x=temp,i=-1)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
)->Float[Tensor,"... queries d_v"]:
    activation_score = calculate_activation_score(Q,K,mask)
    return activation_score @ V

def create_linear(in_features,out_features,weight):
    if weight is None:
        return Linear(in_features,out_features)
    return Linear.from_pretrained(in_features,out_features,weight)

class multihead_self_attention(nn.Module):
    def __init__(self,d_model,num_heads,max_seq_len=None,theta=None,q=None,k=None,v=None,o=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model / self.num_heads
        self.d_v = self.d_model / self.num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = None
        if self.max_seq_len and self.theta:
            self.rope = RoPE(self.theta,self.d_k,self.max_seq_len)
        self.q_proj_weight = create_linear(self.d_model,self.d_model,q)
        self.k_proj_weight = create_linear(self.d_model,self.d_model,k)
        self.v_proj_weight = create_linear(self.d_model,self.d_model,v)
        self.o_proj_weight = create_linear(self.d_model,self.d_model,o)
    
    def forward(self,x):
        q = self.q_proj_weight.forward(x)
        k = self.k_proj_weight.forward(x)
        v = self.v_proj_weight.forward(x)

        q,k,v = map(
            lambda t: rearrange(t,'... l (h d) -> ... h l d',h=self.num_heads),
            (q,k,v)
        )
        l,s = q.shape[-2:]
        causal_mask = ~torch.triu(
            torch.ones(l,l,dtype=torch.bool,device=q.device),
            diagonal=1
        )
        # add rope to q,k
        if self.rope:
            q = self.rope.forward(q,torch.arange(l))
            k = self.rope.forward(k,torch.arange(l))


        attention_score = scaled_dot_product_attention(q,k,v,causal_mask)
        multihead_attention_score = rearrange(attention_score,'... h l d -> ... l (h d)')
        return self.o_proj_weight.forward(multihead_attention_score)


        
