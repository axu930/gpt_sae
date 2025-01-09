import os
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset 

#This uses tiktoken (for GPT2)
#For Llama need to retool to use sentencepiece

local_dir = 'fineweb-edu-10b' # Local directory for storage
remote_name = 'sample-10BT' # HF file name
shard_size = int(1e8) #100m tokens per shard
nprocs = max(1, os.cpu_count() // 2)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    toks = [eot]
    toks.extend(enc.encode_ordinary(doc["text"]))
    toks_np = np.array(toks)
    assert (0 <= toks_np).all() and (toks_np < 2**16).all(), "token dictionary too large for uint16"
    toks_np_uint16 = toks_np.astype(np.uint16)
    return toks_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == '__main__':
    print('Starting extraction')
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0

        for tokens in pool.imap(tokenize, fw, chunksize=8):
            # If there's space left, just append
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)

            # Otherwise split into new shard
            else:
                filename = os.path.join(DATA_CACHE_DIR, f"fineweb_shard_{shard_index:06d}")
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                print(f"Shard {shard_index} processed")

                shard_index += 1
                token_count = len(tokens)-remainder
                all_tokens_np[0:token_count] = tokens[remainder:]

        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR,  f"fineweb_shard_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

