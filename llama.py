import argparse  # Importing module for parsing command-line arguments
import glob  # Importing module for pathname pattern expansion
import json  # Importing module for JSON data handling
import time  # Importing module for time-related functions
from dataclasses import dataclass  # Importing dataclass decorator for creating classes with automatic special methods
from pathlib import Path  # Importing module for representing file system paths
from typing import Optional, Tuple  # Importing module for type hinting

import mlx.core as mx  # Importing mlx core module
import mlx.nn as nn  # Importing mlx neural network module
from mlx.utils import tree_unflatten  # Importing tree_unflatten function from mlx.utils module
from tokenizer import ChatFormat, Dialog, Message, Tokenizer

@dataclass
class ModelArgs:
    dim: int  # Dimensionality of the model
    n_layers: int  # Number of layers in the model
    head_dim: int  # Dimensionality of each attention head
    hidden_dim: int  # Dimensionality of the hidden layer
    n_heads: int  # Number of attention heads
    n_kv_heads: int  # Number of key-value attention heads
    norm_eps: float  # Epsilon value for layer normalization
    vocab_size: int  # Vocabulary size
    rope_theta: float  # Theta value for Relative Positional Encoding (RoPE)
    rope_traditional: bool = True  # Whether to use traditional RoPE


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Store the model arguments within the class instance for easy access.
        self.args = args

        # Number of attention heads and number of key/value heads are specified in the arguments.
        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        # Calculate the number of times the key/value pairs will be repeated across the attention heads.
        self.repeats = self.n_heads // self.n_kv_heads # -> Grouped Query Attention

        # Scale factor for query-key scores, derived from the dimensionality of the head.
        self.scale = self.args.head_dim ** -0.5

        # Define linear transformations for the query, key, and value projections.
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        # Define the linear transformation for the output of the attention mechanism.
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        # Initialize a Rotary Position Embedding (RoPE) module for positional encoding.
        self.rope = nn.RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass of the attention layer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            mask: Optional mask tensor indicating which elements to attend or ignore.
            cache: Optional tuple of cache tensors from previous computations.

        Returns:
            Tuple of output tensor and cache tensor.
        """
        # Extract batch size (B), sequence length (L), and embedding dimension (D) from the input shape.
        B, L, D = x.shape

        # Apply linear transformations to get raw queries, keys, and values.
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Reshape and transpose queries, keys, and values for multi-head attention processing.
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a): # -> Grouped Query Attention 
            # Function to replicate key/value tensors to match the number of attention heads.
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        # Repeat keys and values as needed for attention computation across all heads.
        keys, values = map(repeat, (keys, values))

        if cache is not None:
            # If cache is provided, update queries, keys, and values with cached values.
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            # Apply positional encoding to queries and keys using RoPE.
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Compute attention scores by scaling dot-product of queries and keys
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            # Apply the mask to the scores if provided.
            scores += mask
        # Softmax normalization of the scores to obtain attention weights.
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        # Compute the output by applying the attention weights to the values.
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        # Apply the final linear transformation to the aggregated output.
        return self.wo(output), (keys, values)


# Define a class for the feedforward layer of a neural network model.
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # Initialize the first linear layer with input dimension `dim` and output dimension `hidden_dim`.
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        # Initialize the second linear layer that maps back from `hidden_dim` to `dim`.
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        # An additional third linear layer for a custom operation in the forward pass.
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)


    def __call__(self, x) -> mx.array:
        """
        Performs the forward pass through the feedforward network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after processing through two linear layers and a non-linearity.
        """
        # Apply the first linear transformation followed by a SiLU activation, 
        # then element-wise multiply by the output of the third linear layer applied to input,
        # and finally apply the second linear transformation.
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# Define a class for a single block of a Transformer model, incorporating attention and feedforward layers.
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Number of attention heads.
        self.n_heads = args.n_heads
        # Dimension of input and output.
        self.dim = args.dim
        # Initialize the attention mechanism with the given arguments.
        self.attention = Attention(args)
        # Initialize the feedforward network with the given arguments.
        self.feed_forward = FeedForward(args=args)
        # Layer normalization applied before the attention and after the feedforward layer.
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        # Store the arguments for potential future use.
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Performs the forward pass through the Transformer block.
        
        Args:
            x: Input tensor.
            mask: Optional mask tensor for selective attention.
            cache: Optional cache from previous attention operations for efficiency in decoding tasks.
            
        Returns:
            A tuple of the output tensor and the updated cache.
        """
        # Apply layer normalization, then attention. `cache` is updated/used if provided.
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        # Add the output of the attention layer to the original input (residual connection).
        h = x + r
        # Apply feedforward network to the layer normalized result of the previous step.
        r = self.feed_forward(self.ffn_norm(h))
        # Add the output of the feedforward layer to the result of the residual connection.
        out = h + r
        # Return the output and cache for use in subsequent layers or blocks
        return out, cache


# Defining a class 'Llama' that extends mlx nn.Module, making it a custom neural network module.
class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        # Calling the constructor of the superclass nn.Module.
        super().__init__()  
        # Storing the model arguments passed during initialization.
        self.args = args  
        # Setting the vocabulary size from the arguments.
        self.vocab_size = args.vocab_size  
        # Creating token embeddings.
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)  
        # Initializing transformer blocks as layers according to the number specified in args.n_layers.
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        # RMS normalization layer to normalize the activations of the network.
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        # Linear layer to produce output logits; does not use bias.
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)


    def __call__(self, x):
        """
        Forward pass of the Llama model. It defines how the input tensor 'x' flows through the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Creating an additive causal mask for the input sequence to prevent attention to future tokens.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)  # Ensuring the mask is the same type as embeddings.
        
        x = self.tok_embeddings(x)  # Passing input through token embeddings.
        for l in self.layers:  # Iterating through each Transformer block.
            x, _ = l(x, mask)  # Forward pass through the block with the mask applied.
        x = self.norm(x)  # Applying normalization.
        return self.output(x)  # Returning the final output logits.

    def generate(self, x, temp=1.0):
        """
        Generator function for text generation.
        Generates text by sampling from the distribution provided by the model outputs.

        Args:
            x: Input tensor.
            temp: Sampling temperature.

        Yields:
            Generated tokens.
        """
        def sample(logits):
            # Sampling function to choose output tokens from the logits.
            if temp == 0:
                # If temperature is 0, use argmax for deterministic output.
                return mx.argmax(logits, axis=-1)
            else:
                # Otherwise, sample categorically using the provided temperature to scale logits.
                return mx.random.categorical(logits * (1 / temp))

        cache = []  # Initializing cache for storing intermediate outputs for efficient generation.

        # Creating an additive causal mask for the input sequence as in the forward pass.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)
        #mask = None
        print('mask: ', mask)

        # First we process the prompt x the same way as in __call__ but save the caches in cache
        x = self.tok_embeddings(x)  # Embedding the input tokens.
        for l in self.layers:  # Passing input through each Transformer block.
            x, c = l(x, mask=mask)
            cache.append(c)  # Caching the intermediate outputs for later use.
        x = self.norm(x)
        y = self.output(x[:, -1])  # Getting logits for the last token.
        y = sample(y)  # Sampling a token from the logits.

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y  # Yielding the first generated token.

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None] # Adding a sequence length dimension for the next input.

            x = self.tok_embeddings(x)
            for i in range(len(cache)): # Iterating through cached outputs for efficient generation.
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = sample(self.output(x[:, -1])) # Sampling the next token.

            yield y # Yielding the next generated token in a loop.


def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def generate(args):
    """
    Function to generate text using the Llama model.

    Args:
        args: Command-line arguments.

    Returns:
        None.
    """
    # Waits for the user to press Enter to start the generation process
    input("Press enter to start generation")
    print("------")
    # Prints the prompt that the model will use for generating text
    print('Prompt: ', args.prompt)
    # Converts the prompt into a numerical array format expected by the model, including a beginning-of-sentence token
    #x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(args.prompt)])
    print("bos.id: ", tokenizer.bos_id)
    print('Decoded BOS_ID:', tokenizer.decode([tokenizer.bos_id]))
    print("tokenized prompt: ", tokenizer.encode(args.prompt, bos=True, eos=False))
    #x = mx.array([tokenizer.encode(args.prompt, bos=True, eos=False)]) # text completion
    x = mx.array([ChatFormat(tokenizer).encode_dialog_prompt(x) for x in [ 
        [{"role": "user", "content": args.prompt}]]]) # chat completion
    # Define stop token
    stop_tokens = list(tokenizer.stop_tokens)
    print('Stop Tokens:', stop_tokens)
    skip = 0  # Initializes a counter to keep track of characters already printed to avoid duplicate printing
    prompt_processing = None  # Initializes a variable to hold the time taken for processing the prompt
    tokens = []  # Initializes a list to store generated tokens
    start = tic()  # Starts a timer to measure the duration of text generation

    for token in model.generate(x, args.temp):
        tokens.append(token) # Appends each generated token to the list

        if len(tokens) == 1:
            # After the first token is generated, evaluates the computation time taken to process the prompt
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start) # Records the prompt processing time

        if len(tokens) >= args.max_tokens:
            # Stops generation if the number of generated tokens reaches the specified maximum
            break

        if token in stop_tokens:
            # Stops generation if token is a stop token
            break

        elif (len(tokens) % args.write_every) == 0:
            # Periodically evaluates the tokens and prints the generated text so far. 
            # This happens every 'args.write_every' tokens.
            mx.eval(tokens)  # Evaluates all tokens generated so far
            #print('encoded tokens: ', tokens) # Prints still encoded tokens
            s = tokenizer.decode([t.item() for t in tokens])  # Decodes tokens into text
            print(s[skip:], end="", flush=True)  # Prints the newly generated text since the last print
            skip = len(s)  # Updates the counter to track characters already printed

    # After generation is complete, evaluates any remaining tokens
    mx.eval(tokens)
    full_gen = toc("Full generation", start)  # Records the total time taken for generation
    s = tokenizer.decode([t.item() for t in tokens])  # Decodes all tokens into text
    print(s[skip:], flush=True)  # Prints the final piece of generated text. It means the last token
    print("------")
    print(prompt_processing)  # Prints the time taken for prompt processing
    print(full_gen)  # Prints the total time taken for the text generation

def sanitize_config(config, weights):
    """
    Sanitizes the model configuration by populating missing values.

    Args:
        config: Model configuration dictionary.
        weights: Weights dictionary.

    Returns:
        Sanitized model configuration dictionary.
    """
    # Removes the 'model_type' key from the config dictionary if it exists, as it's not needed further.
    config.pop("model_type", None)

    # Retrieves the number of heads from the config.
    n_heads = config["n_heads"]

    # If 'n_kv_heads' (number of key/value heads) is not specified, it's set equal to the number of heads.
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads

    # If 'head_dim' (dimension of each head) is not specified, it calculates it by dividing the model dimension
    # by the number of heads.
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads

    # If 'hidden_dim' (dimension of the hidden layer) is not specified, it sets it to the shape of the first layer's
    # feed-forward network's first weight dimension from the weights dictionary.
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]

    # Checks if 'vocab_size' is not set or is set to an invalid value (-1); if so, it sets it to the size of the
    # vocabulary as inferred from the shape of the 'output.weight' in the weights dictionary.
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]

    # If 'rope_theta' (a specific configuration parameter, perhaps for a positional encoding scheme) is not set,
    # it defaults to 10000.
    if "rope_theta" not in config:
        config["rope_theta"] = 10000

    # Removes any unused configuration keys that are not required for the current model configuration.
    # Here, 'multiple_of' and 'ffn_dim_multiplier' are explicitly removed from the config.
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)

    # Returns the sanitized configuration dictionary.
    return config


def load_model(model_path):
    """
    Load the Llama model from the given path.

    Args:
        model_path: Path to the model weights and tokenizer.

    Returns:
        Loaded Llama model and tokenizer.
    """
    # Convert model_path to a Path object to utilize pathlib's methods
    model_path = Path(model_path)

    # Define the path to the unsharded (single file) weights
    unsharded_weights_path = Path(model_path / "weights.npz")
    # Check if the unsharded weights file exists
    if unsharded_weights_path.is_file():
        # If the file exists, print an informative message
        print("[INFO] Loading model from {}.".format(unsharded_weights_path))
        # Load the weights from the file
        weights = mx.load(str(unsharded_weights_path))
        #print('weights layers.8.attention.wv.weight:', weights['layers.8.attention.wv.weight'])
        #print(len(weights.items()))
        #print('List of weights:\n', weights.keys())
    else:
        # If the unsharded weights do not exist, handle sharded weights
        # Define the glob pattern for sharded weights files
        sharded_weights_glob = str(model_path / "weights.*.npz")
        # Use glob to find all files matching the pattern
        weight_files = glob.glob(sharded_weights_glob)
        # Print an informative message about loading sharded weights
        print("[INFO] Loading model from {}.".format(sharded_weights_glob))

        # If no weight files are found, raise an error
        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found in {}".format(model_path))

        # Initialize an empty dictionary to hold the weights
        weights = {}
        # Load weights from each file found and update the weights dictionary
        for wf in weight_files:
            weights.update(mx.load(wf).items())
        print('List of weights:\n', weights.keys())

    # Load the model configuration from config.json
    with open(model_path / "config.json", "r") as f:
        # Read and sanitize the configuration, and handle quantization separately
        config = sanitize_config(json.loads(f.read()), weights)
        print('config:', config)
        print('----------------------------')
        quantization = config.pop("quantization", None)
    # Initialize the model with the configuration
    model = Llama(ModelArgs(**config))
    # print model structure
    print('Model Architecture:\n', model)
    # If quantization configuration is present, apply quantization to the model
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    # Update the model weights
    #print('parameters: ', model.parameters()) # randomly created paramaters before update with model parameters
    model.update(tree_unflatten(list(weights.items()))) # Replace the parameters of this Module with the provided ones in the dict of dicts and lists.
    # Load the tokenizer from the tokenizer.model file
    print('#######################')
    tokenizer = Tokenizer(model_path=str(model_path / "tokenizer.model"))
    # Return both the loaded model and tokenizer
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama inference script")
    parser.add_argument(
        "--model-path",
        help="Path to the model weights and tokenizer",
        default="mlx_model",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model. Ignored when --few-shot is provided.",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path)
    print('args: ', args)
    generate(args)
