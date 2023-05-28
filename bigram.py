"""Bigram implementation"""

from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F


BATCH_SIZE = 32
BLOCK_SIZE = 8
LOSS_RATE = 1e-3
EVAL_INTERVAL = 300
EVAL_ITERATIONS = 200
LEARNING_RATE = 1e-2
MAX_ITERS = 3000
MAX_TOKENS = 1000
N_EMBED = 32

#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")


def read_corpus() -> str:
    """Read the data corpus."""
    with open('data/corpus.txt') as f:
        return f.read()
    

class Tokenizer(ABC):
    """Tokenizer base class"""

    vocab_size: int

    @abstractmethod
    def encode(self, input: str) -> list[Union[int, str]]:
        """Tokenize."""

    @abstractmethod
    def decode(self, input: list[Union[int, str]]) -> str:
        """Detokenize."""


class SimpleTokenizer(Tokenizer):

    def __init__(self, chars: list[str]) -> None:
        """Tokenizer based on an input character set."""
        self.stoi = { ch: i for i, ch in enumerate(chars) }
        self.itos = { i: ch for i, ch in enumerate(chars) }
        self.vocab_size = len(chars)

    def encode(self, input: str) -> list[int]:
        """Tokenize."""
        return [ self.stoi[c] for c in input ]

    def decode(self, input: list[int]) -> str:
        """Detokenize."""
        return "".join([ self.itos[i] for i in input ])


class BigramLanguageModel(nn.Module):
    """A Bigram Language Model.
    
    A Bigram language model predicts the probabilty of a word sequence based
    on the previous word.
    """

    def __init__(self, tokenizer: Tokenizer):
        """Initialize BigramLanguageModel."""
        super().__init__()
        self.tokenizer = tokenizer
        self.token_embedding_table = nn.Embedding(tokenizer.vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, tokenizer.vocab_size)

    def forward(self, idx: torch.tensor, targets: Union[torch.tensor, None] = None) -> tuple[torch.tensor, Union[torch.tensor, None]]:
        """..."""
        B, T = idx.shape
    
        # Pytorch will arrange into a Batch (batch sizee), Time (block size), Channel (vocab size)
        # (B, T, C)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # x holds the token embeddings as well as the positions they occur
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss: torch.tensor | None = None
        if targets is not None:
            B, T, C = logits.shape
            # Convert the array to a 2d array, stretched out
            logits = logits.view(B*T, C) # (B, T, C)
            targets = targets.view(B*T)

            # Compute the loss using negative log likelihood loss comparing the prediction (logits)
            # to targets.
            # This wants a B, C, T
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx: torch.tensor, max_new_tokens: int) -> torch.tensor:
        """Append tokens to the end of the sequence.
        
        The idx is the sequence which acts as the input, and we append tokens
        to this and it becomes the output sequence and return value.
        """
        # idx is (B, T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # Apply softmax to get probabilities of each next output in the
            # vocabulary. As a reminder, the C dimension is the vocabulary
            # size and we're doing this in B batches at a time.
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution (making preductions)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append prediction to the running output sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def generate_text(self, max_new_tokens: int) -> str:
        """Generate text."""
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx = self.generate(idx, max_new_tokens=max_new_tokens)
        single_batch = idx[0]
        return self.tokenizer.decode(single_batch.tolist())
    

    def estimate_batch_loss(self, split: torch.tensor) -> float:
        """Estimate the losses for a split of data."""
        losses = torch.zeros(EVAL_ITERATIONS)
        for k in range(EVAL_ITERATIONS):
            xb, yb = get_batch(split)
            _, loss = self(xb, yb)
            losses[k] = loss.item()
        return losses.mean()


def get_batch(data: torch.tensor) -> (torch.tensor, torch.tensor):
    """Generates batch_size inputs (row) at a time each of block_size examples."""
    # Generate a random offset into the training set
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # Inputs
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    # Targets
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    """Main entry point."""

    #torch.manual_seed(1337)


    content = read_corpus()
    chars = sorted(list(set(content)))

    simple = SimpleTokenizer(chars)
    data = torch.tensor(simple.encode(content), dtype=torch.long, device=device)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(simple)
    model = model.to(device)
    
    # AdamW An advanced and much better optimizer than SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=LOSS_RATE)

    # Train
    for step in range(MAX_ITERS):

        if step % EVAL_INTERVAL == 0:
            # Evaluate progress
            with torch.no_grad():
                model.eval()
                train_loss = model.estimate_batch_loss(train_data)
                val_loss = model.estimate_batch_loss(val_data)
                print(f"Step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                model.train()
        
        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())


    print(model.generate_text(MAX_TOKENS))


if __name__ == "__main__":
    main()