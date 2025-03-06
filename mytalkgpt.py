import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import os
from tqdm import tqdm

class GPTConfig:
    """Configuration class for GPT model parameters"""
    def __init__(
        self,
        vocab_size=50257,         # Larger vocabulary (GPT-2 size)
        context_length=1024,      # Maximum sequence length
        embedding_dim=768,        # Hidden dimension
        num_layers=12,            # Number of transformer layers
        num_heads=12,             # Number of attention heads
        dropout=0.1,              # Dropout probability
        layer_norm_epsilon=1e-5,  # Layer norm epsilon
        initializer_range=0.02,   # Weight initialization range
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        assert self.head_dim * config.num_heads == config.embedding_dim, "embedding_dim must be divisible by num_heads"
        
        # Combined projection for query, key, value
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Create causal mask once
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("mask", mask.view(1, 1, config.context_length, config.context_length))
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, 3 * embedding_dim)
        qkv = self.c_attn(x)
        
        # Split into query, key, value and reshape
        q, k, v = qkv.chunk(3, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale dot-product attention with causal mask
        # (batch_size, num_heads, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        
        # Output projection
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.c_proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # GELU activation instead of ReLU
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """Complete GPT model with proper training and generation"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.wpe = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, position_ids=None):
        batch_size, seq_len = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get token and position embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        x = self.drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = F.linear(x, self.wte.weight)
        
        return logits


class TextDataset(Dataset):
    """Dataset for training GPT models on text data with tokenization caching and progress"""
    def __init__(self, data_path, tokenizer, context_length, cache_path="tokens_cache.npy", chunk_size=1024*1024):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.cache_path = cache_path
        
        # Check if tokenized cache exists
        if os.path.exists(cache_path):
            print("Loading tokenized data from cache...")
            self.tokens = np.load(cache_path, allow_pickle=True)
            print("Tokenized data loaded from cache.")
        else:
            print("No cache found. Tokenizing data...")
            file_size = os.path.getsize(data_path)
            tokens = []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar:
                    while True:
                        chunk = f.read(chunk_size)  # Read 1 MB at a time
                        if not chunk:
                            break
                        # Split chunk into smaller pieces <= tokenizer's max_length (1024)
                        for i in range(0, len(chunk), 1024):
                            sub_chunk = chunk[i:i + 1024]
                            if sub_chunk:
                                tokens.extend(tokenizer.encode(sub_chunk, max_length=1024, truncation=True))
                        pbar.update(len(chunk.encode('utf-8')))  # Update by bytes processed
            
            self.tokens = tokens
            print("Tokenization complete. Saving to cache...")
            np.save(cache_path, self.tokens)
            print(f"Tokenized data saved to {cache_path}")
        
    def __len__(self):
        return len(self.tokens) - self.context_length
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train(model, dataloader, optimizer, scheduler, device, max_epochs=50, target_loss=None, patience=3):
    """Training loop with early stopping based on loss monitoring"""
    model.train()
    
    # Track loss history and early stopping
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}") as pbar:
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits = model(x)
                logits = logits.view(-1, model.config.vocab_size)
                y = y.view(-1)
                loss = F.cross_entropy(logits, y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
        
        # Compute and report average epoch loss
        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1} loss: {epoch_loss}")
        
        # Check stopping criteria
        if target_loss is not None and epoch_loss <= target_loss:
            print(f"Target loss {target_loss} reached at epoch {epoch+1}")
            break
        
        # Early stopping based on patience
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0  # Reset if we improve
        else:
            patience_counter += 1
            print(f"No improvement in loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Report training completion
    if epoch == max_epochs - 1:
        print(f"Completed maximum {max_epochs} epochs")
    
    return loss_history


def generate(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40, device="cuda"):
    """Text generation with top-k sampling and temperature"""
    model.eval()
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Only process the last context_length tokens
            if input_ids.size(1) > model.config.context_length:
                input_ids = input_ids[:, -model.config.context_length:]
            
            # Forward pass
            logits = model(input_ids)
            
            # Get the logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            
            # Create a mask for top-k
            mask = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_logits)
            
            # Sample from the filtered distribution
            probs = F.softmax(mask, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add token to sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Check if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated tokens
    output = tokenizer.decode(input_ids[0].tolist())
    
    return output


def main():
    print("Starting the program...")
    print("")

    # Load pre-trained tokenizer
    print("Loading pre-trained tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Tokenizer loaded.")
    print("")

    # Configure and initialize model
    print("Configuring and initializing the model...")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=512,
        embedding_dim=384,
        num_layers=6,
        num_heads=6
    )
    model = GPTModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model configured, initialized, and moved to device:", device)
    print("")

    # Check for existing trained model
    model_path = "my_talk_gpt_model.pt"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print("Trained model loaded successfully.")
        print("")
    else:
        # Create dataset and dataloader
        print("No trained model found. Creating dataset and dataloader...")

        # Replace the text file here with your data
        dataset = TextDataset("replace_this_with_your_dataset.txt", tokenizer, config.context_length, cache_path="tokens_cache.npy")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print("Dataset and dataloader created. Dataset size:", len(dataset), "samples")
        print("")

        # Initialize optimizer and scheduler
        print("Initializing optimizer and scheduler...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader) * 3)
        print("Optimizer and scheduler initialized.")
        print("")

        # Train model with loss monitoring
        print("Training is now starting...")
        loss_history = train(model, dataloader, optimizer, scheduler, device, 
                             max_epochs=50, target_loss=4.0, patience=3)
        print("Training is complete!")
        print("Final loss history:", loss_history)
        print("")

        # Save model
        print("Saving model...")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")
        print("")

    # Interactive text generation
    while True:
        print("Please enter a prompt for text generation (or type 'exit' to quit):")
        user_prompt = input("> ").strip()
        if user_prompt.lower() == "exit":
            break
        
        if not user_prompt:
            print("Prompt cannot be empty. Please try again.")
            print("")
            continue

        print("Generating text...")
        print("")
        generated_text = generate(model, tokenizer, user_prompt, max_length=200, device=device)
        print("Text generation completed. Outputting text...")
        print("")
        print(generated_text)
        print("")
        print("Text generation successful.")
        print("")

    print("Program is complete and ending now.")
    print("")

if __name__ == "__main__":
    main()
