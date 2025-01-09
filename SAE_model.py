import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------

class SAE(nn.Module):
    
    def __init__(self, config, hidden_dim, layer_num = None):
        super().__init__()
        self.layer_num = None
        if layer_num:
            self.layer_num = layer_num

        self.enc = nn.Linear(config.n_embd, hidden_dim, bias = True)
        self.dec = nn.Linear(hidden_dim, config.n_embd, bias = True)


    def forward(self, x, targets = None):
        hidden_activation = F.silu(self.enc(x - self.dec.bias))
        out = self.dec(hidden_activation)
        loss = None

        if targets != None:
            loss = F.mse_loss(out, targets) + F.l1_loss(hidden_activation, torch.zeros_like(hidden_activation)) - 1

        return out, hidden_activation, loss 
    
    def resample(self):
        return NotImplementedError


