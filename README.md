<<<<<<< HEAD
# Llama 3 8B Instruct from Scratch

## Overview

This repository hosts a custom implementation of the "Llama 3 8B Instruct" model using the MLX framework, designed specifically for Apple's silicon, ensuring optimal performance on Apple hardware. Additionally, it integrates a tokenizer created by Meta. The code is restructured and heavily commented to facilitate easy understanding of the key parts of the architecture.

The implementation utilizes the robust MLX library, distinguished for its efficiency and flexibility in handling machine learning tasks on Apple silicon. This compatibility ensures that users can leverage the full capabilities of their hardware for machine learning applications.

## Features

- **RMS-Normalization**: RMSNorm is a simplification of the original layer normalization (LayerNorm). LayerNorm is a regularization technique that might handle the internal covariate shift issue so as to stabilize the layer activations and improve model convergence. It has been proved quite successful in LLaMA 3.
- **Activation Function:** LLaMA 3 uses the SwiGLU activation function instead of ReLU, leading to improved training performance.
- **Rotary Positional Embeddings (RoPE):** Inspired by the GPT-Neo-X project, LLaMA 3 incorporates rotary positional embeddings at each layer, enhancing the model's positional understanding.
- **Increased Context Length and Grouped-Query Attention (GQA)** LLaMA 3 model has a doubled context window (from 4096 to 8192 tokens) and employs grouped-query attention. This allows for better processing of long documents, chat histories, and summarization tasks.
- **Optimized for Apple Silicon:** Fully integrates with the MLX framework, exploiting the performance optimizations available on Apple silicon for superior model efficiency.

## Usage

### Setup

To utilize this model, clone the repository and install the necessary dependencies as outlined in the `requirements.txt` file (not included in this readme but should be part of your repository setup).

### Running the Model

Execute the model via the command line with various configurable options:

- `--model-path`: Path to the model weights and tokenizer.
- `--prompt`: The initial text to kickstart the generation process.
- `--max-tokens`: The maximum number of tokens to generate.
- `--write-every`: The interval of token detokenization to check the output during generation.
- `--temp`: The sampling temperature, which affects the randomness of the output.
- `--seed`: The seed for the pseudo-random number generator, ensuring reproducibility.

Example command:

```bash
python model_script.py --model-path ./mlx_model --prompt "Hello, world!" --max-tokens 50
```

### Text Generation

The `generate` function manages text generation, orchestrating the model's capabilities to produce text from input prompts. It oversees the conversion of text to tokens, processes through the model, and transforms tokens back to text.

## Tokenizer

The tokenizer is integral for converting text into a format the neural network can process, segmenting language into manageable components and facilitating the model to generate human-like text.

## Contributions

Contributions are welcome! If you have improvements or bug fixes, please submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details (assuming a standard open-source license for demonstration).
=======

>>>>>>> 9113a91b57556bff5c1a74db9ec8aa3d1594e1a1
