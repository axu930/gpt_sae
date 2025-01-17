# gpt_sae
Training SAEs on open source GPTs in the style of Anthropic's monosemanticity. [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)

# Details
- Base model is the GPT2 small model from [https://huggingface.co/docs/transformers/en/model_doc/gpt2](https://huggingface.co/docs/transformers/en/model_doc/gpt2). 
- SAE is trained on the fineweb 10b corpus [https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). Said corpus is obtained and tokenized with get_fineweb.py. 
- Hidden dim for SAE is 8192, trained on the middle layer MLP (layer no. 6).
- SAEs were trained with different loss functions. 
- Learn rate is 1e-4

## L1 SAEs
SAEs trained with the loss function

    loss = mse_loss + l1_loss

in the style of Anthropic's paper. There are pre-encoder and pre-relu biases added.  

## JL SAEs
SAEs trained with the loss function inspired by the Johnson-Lindenstrauss (JL) lemma:

    loss = mse_loss + jl_loss

where jl_loss is defined by

    jl_loss = atanh(A_enc @ A_enc.T).sum

Since all columns of A_enc are unit length, this optimizes the vectors to be 'as orthogonal as possible' from each other. The atanh is so that the objective is convex. A 'warmup' is used on the jl_loss coefficient to keep the vectors spread out in the first 2000 steps of training.

## Hybrid SAEs
Using all 3 different loss functions

    loss = mse_loss + jl_loss

# TODO
- Visualization functs
- Helper functions for steering
- Neuron activation tracking
- Neuron resampling

## Experiments TODO:
- Ablation testing on each paradigm with different loss coefficients
- Compare L1, Johnson-Lindenstrauss, and hybrid trained SAEs
- Ablation testing with encoder/decoder learn rate decoupling / scheduling (decoder learns faster)
- Train SAEs across different layers

## Down the line experiments:
- Other AE architectures: FSQAE, VAE/NVAE (other forms of dictionary learning)
- Experiment with context length
- Experiment with LoRA tuning (see if learned features persist through finetuning)
- Implementation on other GPT models -- LLaMA, Mistral (tokenize fineweb with sentencepiece instead of tiktoken)
