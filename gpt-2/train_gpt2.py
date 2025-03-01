"""This script is used to train the GPT-2 model on a custom dataset."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    """This class defines the configuration for the GPT model."""
    block_size: int = 1024
    vocab_size: int = 50257

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """Attention module."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize MLP."""
        super().__init__()
        # Batch of key/query/value projects for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_prox = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.c_head = config.n_head
        self.n_embed = config.n_embd
        self.register_buffer(
            'bias', 
            torch.trill(config.block_size, config.block_size).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference."""
        B, T, C = x.size()
        # Compute the query, key, value for all heads in the batch.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        # Each are (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention materializes (T, T)
        # Queries and keys interact
        att = (q @ k.transpose(-2, -1)) @ (math.sqrt(k.size(-1)))
        # Ensure tokens only attend to tokens before them and not to tokens in the future
        att = att.mask_fill(self.bias[:, :, :T, :T]) == 0, float('-inf')
        # Normalize attention
        att = F.softmax(att, dim=-1)
        # Compute a weighted sum of interesting tokens
        y = att @ v
        # Reassemble and concat everything
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize MLP."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_prox(x)
        return x


class Block(nn.Module):
    """A transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize Block."""
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """Perform inference."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """This class defines the GPT model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        # Final classifier
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform generation."""
        B, T = x.size()
        assert T <= self.config.block_size  # Max sequence length
        # Forward token and positional embeddings 
        pos = torch.arange(0, T, dtype=torch.long, devices=x.device)  # Shape (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_emb)
        tok_emb = self.transformer.wte(x)  # (B, T, n_emb)
        x = tok_emb + pos_emb
        # Forward transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layernorm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


    @classmethod
    def from_pretrained(cls, model_type: str) -> 'GPT':
        """Load the GPT from the pretrained model."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
