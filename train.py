"""
Training script for Multi-Tower MoE model on TinyShakespeare dataset.
Pure PyTorch implementation (no HuggingFace).
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import MoEModel


class CharDataset(Dataset):
    """
    Character-level dataset for language modeling.
    """
    def __init__(self, text: str, seq_len: int = 128, batch_size: int = 32):
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode text to indices
        self.data = [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
        self.total_len = len(self.data)

    def __len__(self):
        # Number of batches
        return (self.total_len - self.seq_len) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch_indices = []
        for _ in range(self.batch_size):
            seq_start = start + _ * self.seq_len
            if seq_start + self.seq_len + 1 < self.total_len:
                seq = self.data[seq_start:seq_start + self.seq_len]
                target = self.data[seq_start + 1:seq_start + self.seq_len + 1]
                batch_indices.append((seq, target))
            else:
                break

        if len(batch_indices) == 0:
            # Fallback
            start = idx * self.seq_len
            if start + self.seq_len + 1 < self.total_len:
                seq = self.data[start:start + self.seq_len]
                target = self.data[start + 1:start + self.seq_len + 1]
                batch_indices.append((seq, target))

        x = torch.tensor([item[0] for item in batch_indices], dtype=torch.long)
        y = torch.tensor([item[1] for item in batch_indices], dtype=torch.long)
        return x, y


def download_tiny_shakespeare():
    """Download or generate TinyShakespeare dataset."""
    data_path = "shakespeare.txt"

    if not os.path.exists(data_path):
        # Generate a small sample if download fails
        sample_text = """Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.

Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventur'd piteous overthrows
Doth with their death bury their parents' strife."""
        with open(data_path, 'w') as f:
            f.write(sample_text)

    with open(data_path, 'r') as f:
        text = f.read()

    return text


def train():
    """Main training loop."""
    # Configuration
    vocab_size = 256  # ASCII characters
    d_model = 64
    d_ff = 256
    num_experts = 8
    top_k = 2
    num_layers = 1

    seq_len = 128
    batch_size = 32
    learning_rate = 1e-3
    max_steps = 5000
    grad_clip = 1.0
    checkpoint_freq = 500

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    text = download_tiny_shakespeare()
    dataset = CharDataset(text, seq_len, batch_size)

    if len(dataset) == 0:
        print("Dataset too small, creating synthetic data...")
        synthetic_text = "AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXXYYZZ0123456789 " * 10000
        dataset = CharDataset(synthetic_text, seq_len, batch_size)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset size: {len(dataset)} batches")
    print(f"Vocabulary size: {dataset.vocab_size}")

    # Initialize model
    model = MoEModel(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        num_layers=num_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training state
    model.train()
    checkpoint_path = "checkpoint.pt"

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # x: [batch, seq_len]
        # y: [batch, seq_len]

        optimizer.zero_grad()

        # Forward pass
        logits, aux_loss = model(x)
        # logits: [batch, seq_len, vocab_size]

        # Compute task loss (cross-entropy over flattened sequence)
        batch_size, seq_len, _ = logits.shape
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

        # Total loss
        total_loss = loss + aux_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Step optimizer
        optimizer.step()

        # Logging
        if step % 10 == 0:
            print(f"Step {step}: Loss={total_loss.item():.4f}, AuxLoss={aux_loss.item():.4f}")

        # Checkpointing
        if step % checkpoint_freq == 0 and step > 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step}")

        if step >= max_steps:
            break

    # Final save
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
    }, checkpoint_path)

    print(f"Training complete! Final step: {step}")
    print(f"Final loss: {total_loss.item():.4f}")
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
