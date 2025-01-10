# gpt_sae
Training SAEs on open source GPTs in the style of Anthropic's monosemanticity. [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)

# Details
- Models are the GPT2 small model from [https://huggingface.co/docs/transformers/en/model_doc/gpt2](https://huggingface.co/docs/transformers/en/model_doc/gpt2). 
- SAE is trained on the fineweb 10b corpus [https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- Hidden dim for SAE is 8192, trained on the middle layer MLP

# TODO
- Visualization functs
- Helper functions for steering
- Neuron resampling

Down the line:
- Experiment with context length
- Experiment with other loss functions (to force more sparsity)
- Experiment with LoRA tuning (see if monosemantic features persist)
- Experiment with using early exit criteria to train SAEs
- Other AE architectures: FSQAE, VAE/NVAE (other forms of dictionary learning)
- Other GPT models -- LLaMA, Mistral (tokenize fineweb with sentencepiece instead of tiktoken)
