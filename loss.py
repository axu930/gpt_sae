import torch
import torch.nn.functional as F

from configs import device, SAE_Config, sparsity_lambda

"""
Loss functions expect 2d tensors in the following shapes:
[batch, inputs] for reconstruction loss  
[batch, activations] for L1 loss (basis pursuit)
[sae enc in, sae enc out] for JL term loss

Flatten tensors from [B,T,C] into [B*T, C] before passing to loss function
"""

def autoencoder_L1_loss(out, targets, activations): 
    return (
        F.mse_loss(out, targets) 
        + sparsity_lambda * normalized_L1_loss(activations)
    )

def autoencoder_JL_loss(out, targets, w_enc):
    return (
        F.mse_loss(out, targets) 
        + sparsity_lambda * normalized_JL_loss(w_enc)
    )


def normalized_L1_loss(activations):
    return activations.sum(dim=1).mean()

def normalized_JL_loss(w_enc):
    mask = (torch.arange(SAE_Config.h_dim)[:,None] != torch.arange(SAE_Config.h_dim)[None,:]).to(device)
    dot_matrix = F.relu((w_enc @ w_enc.T) * mask)
    return F.l1_loss(dot_matrix, torch.zeros_like(dot_matrix)) * SAE_Config.h_dim
