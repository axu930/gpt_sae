import torch
import torch.nn.functional as F
import math

from configs import device, SAE_Config, L1_lambda, JL_lambda

"""
Loss functions expect 2d tensors in the following shapes:
[batch, inputs] for reconstruction loss  
[batch, activations] for L1 loss (basis pursuit)
[sae enc in, sae enc out] for JL term loss

Flatten tensors from [B,T,C] into [B*T, C] before passing to loss function
"""

def autoencoder_L1_loss(reconstruction_errors: torch.Tensor, 
                        activations: torch.Tensor, 
                        sae_inputs: torch.Tensor,
                        coeff = L1_lambda): 
    return (
        (reconstruction_errors ** 2).mean()
        + coeff * normalized_L1_loss(sae_inputs, activations)
    )

def autoencoder_JL_loss(reconstruction_errors: torch.Tensor, 
                        w_enc: torch.Tensor,
                        coeff = JL_lambda):
    return (
        (reconstruction_errors ** 2).mean()
        + coeff * JL_loss(w_enc, input_dim=reconstruction_errors.shape[1])
    )

def normalized_mse_loss(reconstructon_errors: torch.Tensor,
                        sae_inputs: torch.Tensor, ):
    return (
        (reconstructon_errors ** 2).mean(dim=1) / (sae_inputs ** 2).mean(dim=1)
            ).mean()

def normalized_L1_loss(sae_inputs: torch.Tensor, 
                       activations: torch.Tensor,):
    return (
        activations.sum(dim=1) / sae_inputs.norm(dim=1)
        ).mean()

def JL_loss(w_enc: torch.Tensor, input_dim=1):
    mask = (torch.arange(SAE_Config.h_dim)[:,None] != torch.arange(SAE_Config.h_dim)[None,:]).to(device)
    dot_matrix = F.relu((w_enc @ w_enc.T) * mask)
    return dot_matrix.mean() * math.sqrt(input_dim) # dimension specific normalization term
