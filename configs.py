import os


device = "mps:0"

weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(weights_dir, exist_ok=True)

# tokenized data, 100 shards total
data_dir = os.path.join(os.path.dirname(__file__), 'fineweb-edu-10b')
data_files = os.listdir(data_dir)
shards = [os.path.join(data_dir,shard) for shard in data_files]
val_shards = shards[:5]
train_shards = shards[5:]

# In the loss function, the ratio of reconstruction loss : sparsity loss
L1_lambda = 0.01
JL_lambda = 1

# ----------

class GPT_Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT_medium_Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 24 # number of layers
    n_head: int = 16 # number of heads
    n_embd: int = 1024 # embedding dimension


class SAE_Config:
    h_dim: int = 8192 
    bias: bool = True
    loss_fn: str = 'l1'

class JL_SAE_Config:
    h_dim: int = 8192 
    bias: bool = False
    loss_fn: str = 'jl'
