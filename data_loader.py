import torch
import numpy as np
import random

# assumes raw tiktoken-ized numbers

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader:

    def __init__(self, B, T, shards):
        self.B = B
        self.T = T
        self.shards = shards

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