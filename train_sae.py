import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_model import GPT
from SAE_model import SAE
from data_loader import DataLoader

import loss as L
from configs import *

# ---------

# What are we testing
run_name = 'JL_trial' ##
model_name = 'gpt2-small'

# Model configs
gpt_config = GPT_Config
sae_config = JL_SAE_Config
sae_layer = gpt_config.n_layer // 2 # MLP layer number that we are trying to approximate

load_weights = False

# Training configs
B = 8 #Batch size
T = 1024 #Sequence length
lr = 1e-4 #Learning rate
steps = 5000

# Model init
model = GPT.from_pretrained('gpt2')
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

sae_model = SAE(gpt_config, sae_config)
sae_model.to(device)

# Load in weights 
weight_file = os.path.join(weights_dir, f"{model_name}_{run_name}_layer{sae_layer}.ckpt")
if os.path.exists(weight_file) and load_weights:
    print('Loading in pretrained SAE weights')
    sae_model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))

# ---------


def train():
    optimizer = torch.optim.AdamW(sae_model.parameters(), lr=lr)
    train_loader = DataLoader(B, T, train_shards)
    val_loader = DataLoader(B, T, val_shards)
    print("Starting training")
    t0 = time.time()

    for step in range(steps):
        sae_model.train()
        optimizer.zero_grad()
        x = train_loader.next_batch()
        x = x.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with torch.no_grad():
                _, _, stream = model(x, get_stream=True)
            sae_data = stream[sae_layer]
            sae_in, sae_out = sae_data[0].flatten(start_dim=0,end_dim=1), sae_data[1].flatten(start_dim=0,end_dim=1)
            out, hidden_activation = sae_model(sae_in) 

            if sae_config.loss_fn == 'l1':
                loss = L.autoencoder_L1_loss(out - sae_out, 
                                    hidden_activation,
                                    sae_in,)
            else:
                loss = L.autoencoder_JL_loss(out - sae_out, 
                                    sae_model.enc.weight)

        loss.backward()

        if step % 100 == 0:
            loss = loss.detach()
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_per_sec = B * T * 100 / dt
            t0 = t1
            print(f"step: {step:5d}, train loss: {loss.item():.4f}, mse loss: {((out - sae_out) ** 2).mean()}, dt: {dt:.2f}s , tok/sec: {tokens_per_sec:.2f}")

        # make sure optimizer trajectories are perpendicular to encoder vectors
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                dot = torch.linalg.vecdot(sae_model.enc.weight,sae_model.enc.weight.grad)
                sae_model.enc.weight.grad -= dot.unsqueeze(1) * sae_model.enc.weight
        
        optimizer.step()
        
        #encoder normalization to prevent long term training drift
        with torch.no_grad():
            norms = torch.sqrt(torch.linalg.vecdot(sae_model.enc.weight, sae_model.enc.weight))
            sae_model.enc.weight /= norms.unsqueeze(1)

        if step % 1000 == 0 or step == steps - 1:
            sae_model.eval()
            with torch.no_grad():
                x = val_loader.next_batch()
                x = x.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, _, stream = model(x, get_stream=True)
                    sae_data = stream[sae_layer]
                    sae_in, sae_out = sae_data[0].flatten(start_dim=0,end_dim=1), sae_data[1].flatten(start_dim=0,end_dim=1)
                    out, hidden_activation = sae_model(sae_in) 
                    if sae_config.loss_fn == 'l1':
                        loss = L.autoencoder_L1_loss(out - sae_out, 
                                            hidden_activation,
                                            sae_in,)
                    else:
                        loss = L.autoencoder_JL_loss(out - sae_out, 
                                            sae_model.enc.weight)
                loss = loss.detach()
                print(f"step: {step:5d}, validation loss: {loss.item():.4f}, mse loss: {((out - sae_out) ** 2).mean()}")
            torch.save(sae_model.state_dict(), weight_file)
            print(f"Model weights saved to {weight_file}")
            
    print("Finished training!")

if __name__ == '__main__':
    train()
