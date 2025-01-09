import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from SAE_model import *

device = "mps:0"


# ---------
# assumes raw tiktoken-ized numbers

data_dir = os.path.join(os.path.dirname(__file__), 'fineweb-edu-10b')
data_files = os.listdir(data_dir)

val_shards = data_files[:5]
train_shards = data_files[5:]

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader:

    def __init__(self, B, T, shards):
        self.B = B
        self.T = T
        self.shards = [os.path.join(data_dir, shard) for shard in shards]

        self.current_position = 0
        self.current_shard = random.randint(0,len(shards) - 1)

        self.tokens = load_tokens(self.shards[self.current_shard])        


    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T]
        self.current_position = self.current_position + B*T
        x = buf.view(B, T) # inputs
        
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        
        return x

# ---------
# Train run configs

from gpt2_model import * 

model_name = 'gpt2-small'
run_name = 'trial'
load_weights = True
config = GPT_Config
h_dim = 8192


B = 8 #Batch size
T = 1024 #Sequence length
lr = 1e-4 #Learning rate
sae_layer = config.n_layer - 1 #Layer number of MLP we are trying to approximate
steps = 2000

# Model inits

model = GPT.from_pretrained('gpt2')
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

sae_model = SAE(config, hidden_dim=h_dim)
sae_model.to(device)

# Load in weights 

weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(weights_dir, exist_ok=True)
weight_file = os.path.join(weights_dir, f"{model_name}_{run_name}.ckpt")
if os.path.exists(weight_file) and load_weights:
    sae_model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))


# ---------

import time 
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
            _, _, stream = model(x, get_stream=True)
            sae_data = stream[sae_layer]
            sae_in, sae_out = sae_data[0], sae_data[1]
            _, _, loss = sae_model(sae_in, targets=sae_out) 
        loss.backward()
        optimizer.step()
        del stream
        del sae_data

        if step % 100 == 0:
            loss = loss.detach()
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_per_sec = B * T * 100 / dt
            t0 = t1
            print(f"step: {step:5d}, train loss: {loss.item():.4f}, dt: {dt*1000:.2f}ms , tok/sec: {tokens_per_sec:.2f}")
            

        if step % 1000 == 0 or step == steps - 1:
            sae_model.eval()
            with torch.no_grad():
                x = val_loader.next_batch()
                x = x.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, _, stream = model(x, get_stream=True)
                    sae_data = stream[sae_layer]
                    sae_in, sae_out = sae_data[0], sae_data[1]
                    _, _, loss = sae_model(sae_in, targets=sae_out) 
                loss = loss.detach()
                print(f"step: {step:5d}, validation loss: {loss.item():.4f}")
            torch.save(sae_model.state_dict(), weight_file)
            
    print("Finished training!")


if __name__ == '__main__':
    train()
