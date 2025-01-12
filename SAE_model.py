import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------

class SAE(nn.Module):
    
    def __init__(self, config, sae_config, layer_num = None):
        super().__init__()
        self.layer_num = None
        if layer_num:
            self.layer_num = layer_num

        self.enc = nn.Linear(config.n_embd, sae_config.h_dim, bias = True)
        self.dec = nn.Linear(sae_config.h_dim, config.n_embd, bias = True)

        nn.init.kaiming_uniform_(self.enc.weight, a=0, mode='fan_in', nonlinearity='relu')
        norms = torch.sqrt(torch.linalg.vecdot(self.enc.weight,self.enc.weight))
        with torch.no_grad():
            self.enc.weight /= norms.unsqueeze(1)


    def forward(self, x):
        hidden_activation = F.relu(self.enc(x - self.dec.bias))
        out = self.dec(hidden_activation)
        return out, hidden_activation
    
    def resample(self):
        return NotImplementedError


