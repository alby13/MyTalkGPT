# MyTalkGPT
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

<br>

<img src="https://github.com/alby13/MyTalkGPT/blob/main/running-screenshot.png">

<br>

Train Your Own LLM - Simple! Easy!

Artificial Intelligence training world uses a lot of words. 

LLM Training programs try to reproduce complicated or advanced AI models.

MyTalkGPT is a training program that is easy to understand and simplified from a full production ChatGPT Large Language Model.

<br><br>

### Main Program Functions:

Configuration: Creates a GPTConfig object to store model hyperparameters.

Model Initialization: Initializes the GPTModel.

Device Selection: Determines whether to use a GPU (NVIDIA CUDA) or CPU.

Model Loading: Attempts to load a pre-trained model from a file (my_talk_gpt_model.pt). If you create the model with this program, the program will load it again so you can prompt it for test output.

Training or Loading: If a pre-trained model is not found, trains a new model.

Interactive Generation: Enters a loop that prompts the user for input and generates text based on the prompt.

Model Saving: Saves the trained model to a file.

Error Handling: Will catch a blank input during the prompt and prevent erroring out.

<br>

### How the code was formatted:

Modularity: The code is well-structured, with separate classes and functions for different components (model, attention, training, etc.). This makes the code easier to understand, maintain, and extend.

Configuration Class: Uses a GPTConfig class to manage model hyperparameters, making it easy to change settings.

Comments: The code is well-commented, explaining the purpose of different parts of the code.

Type Hints: Uses type hints (e.g., -> torch.Tensor) to improve code readability and help catch errors.

Use of Libraries: Leverages powerful libraries like torch, transformers, and tqdm for efficient and robust implementation.


In summary, this code provides a complete system for training and using a GPT-like language model, including data loading, preprocessing, training, generation, and interactive use.

<br>

<br>

## How to use/How to Train:

You need to prepare a text file with approximately around 500 megabytes of text data, give or take.

1. Replace <code>replace_this_with_your_dataset.txt</code> with your text file with your training dataset text.

2. Run MyTalkGPT.py

3. Wait for training. First your tokenized file will be saved, then your AI will be trained and saved, then you can enter prompts for the AI to respond to.
   
<br>

<br>

<br>

## What kind of LLM will this training produce?

This code implements a GPT (Generative Pre-trained Transformer) model that is inspired by GPT-2, but it's significantly smaller and less capable than the actual GPT-2. Let's break down the key differences and how close it is:

Similarities to ChatGPT (GPT-2):

    Architecture: The core architecture is very similar. It uses:

        Transformer Blocks: The fundamental building block is a stack of transformer decoder layers, just like GPT-2.

        Multi-Head Self-Attention: The attention mechanism is multi-headed, allowing the model to attend to different parts of the input sequence in parallel.

        Causal Masking: The attention mechanism uses a causal mask, ensuring that the model can only attend to past tokens, preventing it from "cheating" by looking at future tokens during training and generation.

        Feed-Forward Networks: Each transformer block contains a feed-forward network (with GELU activation, which is correct for GPT-2).

        Layer Normalization: Layer normalization is used before the attention and feed-forward layers (pre-norm), which is the modern and more stable configuration. GPT-2 originally used post-norm, but later models (and best practices) switched to pre-norm.

        Token and Position Embeddings: The model uses learned embeddings for both tokens (words) and positions (where the word appears in the sequence).

        Weight Initialization: The code includes a weight initialization scheme similar to GPT-2's.

        Training Loop: The train function implements a standard training loop with backpropagation, gradient clipping, and a learning rate scheduler.

        Generation: The generate function uses top-k sampling, a common technique for generating text with language models. It also correctly handles the context length during generation.

        Tokenizer: It uses the GPT2Tokenizer from the transformers library, which is crucial. This means it's using the same vocabulary and tokenization method as GPT-2, ensuring compatibility at the input/output level.

        Dataset Class: The TextDataset class shows good best practice by using the same tokenizer.

Key Differences and How They Impact Closeness:

    Model Size (The Biggest Difference):

        embedding_dim=384: GPT-2 (small) uses embedding_dim=768. GPT-2 (medium) uses 1024, large uses 1280, and XL uses 1600. The embedding dimension is the size of the hidden state, and it's a major factor in model capacity. This model's embedding dimension is half the size of even the smallest GPT-2.

        num_layers=6: GPT-2 (small) uses num_layers=12. Medium uses 24, large uses 36, and XL uses 48. The number of layers is another huge factor. This model has half the layers of the smallest GPT-2.

        num_heads=6: GPT-2 (small) uses num_heads=12. This is directly tied to the embedding_dim. Since embedding_dim is halved, num_heads is also halved to keep the head dimension consistent.

        context_length = 512 While GPT-2's official context length is 1024, this model's context is reduced to 512. This is important for memory usage.

    Impact: These size differences are enormous. The model capacity (its ability to learn complex patterns in the data) is drastically reduced compared to even the smallest GPT-2 model. It will be much less coherent, have a much poorer understanding of language, and struggle with longer-range dependencies. A rough estimate: the parameter count of this model is about 1/8 the size of GPT-2 small.

    Training Data and Time:

        The code includes a train function, but it's configured for a relatively small number of epochs (max_epochs=50) and a small batch size (batch_size=4). GPT-2 was trained for a vastly longer time on a massive dataset (WebText, which is many gigabytes of text).

        The books_trimmed_500mb.txt file is much smaller than the datasets used to train GPT-2. 500MB is still a reasonable dataset size for this model.

        Impact: Even if the architecture were identical, the limited training data and time would result in a model far less capable than GPT-2. The model will likely overfit to this smaller dataset and won't generalize as well.

    Optimizer and Learning Rate:

        The code uses AdamW with a learning rate of 5e-5 and a cosine annealing scheduler. These are reasonable choices, but the optimal hyperparameters would depend on the dataset and model size. GPT-2's training used carefully tuned learning rate schedules.

        Impact: While the optimizer choice is good, the learning rate might not be optimal. However, the impact is less significant than the model size and training data differences.

    Early Stopping:

        The provided training loop correctly implements early stopping, an important practice.

    Gradient Clipping:

        Implemented, and an important practice for stable training.

Overall Closeness:

The code implements a model that shares the fundamental architecture of GPT-2, but it's a much smaller and less powerful model. It's best described as a "mini-GPT" or a "GPT-2-like" model. It's a good starting point for learning about transformers and language models, but it won't produce results comparable to even the smallest released GPT-2 model. Think of it as a scaled-down, simplified version designed for educational purposes or experimentation on smaller datasets.

In Summary:

    Architecturally: Very similar in terms of the core components (transformer blocks, attention, etc.).

    Capacity: Significantly smaller and less capable due to reduced embedding_dim, num_layers, and num_heads.

    Training: Designed for much shorter training on smaller datasets.

    Performance: Will be far less coherent and knowledgeable than any of the official GPT-2 models.

It's much closer to a "toy" GPT-2 than a production-ready language model. But it's an excellent, well-written, and well-commented codebase for learning and experimentation! The use of the transformers tokenizer and the overall structure are excellent starting points.

<br>

## Information on the training program

1. Core Model Architecture (GPT-like):

    Transformer Decoder: The model is based on the transformer decoder architecture, similar to GPT-2. This means it's designed for autoregressive language modeling (predicting the next word in a sequence).

    Multi-Head Self-Attention:

        Implements the core attention mechanism that allows the model to weigh the importance of different words in the input sequence when predicting the next word.

        Uses multiple attention "heads" to learn different relationships between words.

        Includes a causal mask to prevent the model from "looking ahead" at future tokens.

        Uses combined query, key, and value projections for efficiency.

        Includes attention dropout and residual dropout for regularization.

    Feed-Forward Network (FFN):

        A two-layer fully connected network with a GELU activation function (consistent with GPT-2).

        Expands the dimensionality of the hidden state and then projects it back down.

        Includes dropout for regularization.

    Transformer Blocks:

        Combines the multi-head self-attention and FFN modules into a single block.

        Uses pre-layer normalization (LayerNorm before the attention and FFN).

        Implements residual connections (adding the input to the output of each sub-layer).

    Token Embeddings:

        Learned embeddings that map each word in the vocabulary to a dense vector representation. Uses nn.Embedding.

    Positional Embeddings:

        Learned embeddings that encode the position of each word in the sequence. Uses nn.Embedding. This is crucial for transformers, as they don't inherently have a sense of word order.

    Final Layer Normalization:

        Applies layer normalization to the output of the final transformer block.

    Output Projection:

        Projects the final hidden state to the vocabulary size, producing logits (unnormalized probabilities) for each word.

    Weight Initialization:

        Initializes the weights of the linear and embedding layers with a normal distribution.

        Initializes biases to zero.

        Initializes layer normalization weights to one and biases to zero.

2. Data Handling and Preprocessing:

    GPT2Tokenizer:

        Uses the pre-trained GPT2Tokenizer from the transformers library. This is critical for:

            Ensuring the model uses the same vocabulary as GPT-2.

            Correctly handling special tokens (like end-of-sequence).

            Providing consistent tokenization (splitting words into sub-word units).

    TextDataset Class:

        Handles loading and tokenizing text data.

        Caching: Saves the tokenized data to a file (tokens_cache.npy) to avoid re-tokenizing on subsequent runs. This is a very important optimization for large datasets.

        Chunking: Reads the input file in chunks to handle large files efficiently.

        Sub-Chunking: Tokenizes in sub-chunks that are smaller than the tokenizer maximum length.

        Progress Bar: Uses tqdm to display a progress bar during tokenization, providing visual feedback.

        Creates input (x) and target (y) tensors for training, where y is the shifted version of x.

    DataLoader:

        Uses PyTorch's DataLoader to efficiently batch and shuffle the data during training.

3. Training Process:

    train Function:

        Implements the core training loop.

        Device Handling: Moves the model and data to the specified device (CPU or GPU).

        Forward Pass: Calculates the model's output (logits).

        Loss Calculation: Computes the cross-entropy loss between the predicted logits and the target tokens.

        Backward Pass: Calculates gradients.

        Gradient Clipping: Clips gradients to prevent exploding gradients (using torch.nn.utils.clip_grad_norm_).

        Optimizer Step: Updates the model's weights using the optimizer.

        Scheduler Step Updates the learning rate

        Progress Bar: Displays a progress bar with the current loss using tqdm.

        Early Stopping: Monitors the training loss and stops training if the loss doesn't improve for a specified number of epochs (patience). This prevents overfitting.

        Target Loss: Allows training to stop if a certain loss is met.

        Loss History: Returns the loss history so it can be inspected.

    Optimizer:

        Uses the AdamW optimizer (a variant of Adam with weight decay), a good default choice for training transformers.

    Learning Rate Scheduler:

        Uses a cosine annealing learning rate scheduler (CosineAnnealingLR). This gradually decreases the learning rate over time, which can help improve performance.

4. Text Generation:

    generate Function:

        Implements text generation using the trained model.

        Prompt Handling: Takes a text prompt as input.

        Tokenization: Tokenizes the prompt using the GPT2Tokenizer.

        Iterative Generation: Generates text one token at a time.

        Context Window: Only considers the last context_length tokens for prediction, ensuring the model doesn't exceed its maximum sequence length.

        Temperature Scaling: Divides the logits by a temperature value to control the randomness of the generated text. Higher temperatures lead to more random output.

        Top-k Sampling: Filters the logits to only consider the top k most likely tokens. This prevents the model from generating very low-probability (and often nonsensical) words.

        Multinomial Sampling: Samples the next token from the filtered probability distribution.

        EOS Token Handling: Stops generation if the end-of-sequence (EOS) token is generated.

        Decoding: Decodes the generated tokens back into text using the GPT2Tokenizer.

    Evaluation Mode: Sets the model to evaluation mode (model.eval()) during generation, disabling dropout.

    No Gradient Calculation: Uses torch.no_grad() to disable gradient calculations during generation, saving memory and computation.
