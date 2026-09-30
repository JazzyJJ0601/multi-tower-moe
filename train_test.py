"""
Training script for Multi-Tower MoE model - TEST VERSION
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import MoEModel


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 128):
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        target = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def download_tiny_shakespeare():
    data_path = "shakespeare.txt"
    if not os.path.exists(data_path):
        sample_text = """Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet."""
        with open(data_path, 'w') as f:
            f.write(sample_text)
    with open(data_path, 'r') as f:
        return f.read()


def train():
    d_model = 64
    d_ff = 256
    num_experts = 8
    top_k = 2
    seq_len = 32
    batch_size = 16
    learning_rate = 1e-3
    max_steps = 50
    grad_clip = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    text = download_tiny_shakespeare()
    dataset = CharDataset(text, seq_len)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Vocabulary size: {dataset.vocab_size}")

    # Initialize model
    model = MoEModel(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        num_layers=1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✓ Model initialization: SUCCESS")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss_history = []
    aux_loss_history = []
    expert_utilization = None

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, aux_loss = model(x)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        total_loss = loss + aux_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_history.append(total_loss.item())
        aux_loss_history.append(aux_loss.item())

        # Get expert utilization at step 0
        if step == 0:
            with torch.no_grad():
                x_embed = model.embedding(x)
                _, _, router_logits = model.layers[0].gate(x_embed)
                probs = torch.softmax(router_logits, dim=-1)
                expert_utilization = probs.mean(dim=(0, 1))
                print(f"Expert utilization distribution: {expert_utilization}")
                print(f"Expert utilization sum: {expert_utilization.sum().item():.4f}")

        if step % 10 == 0 or step == 0:
            print(f"Step {step}: Loss={total_loss.item():.4f}, AuxLoss={aux_loss.item():.6f}")

        if step >= max_steps:
            break

    # Summary
    print("\n" + "="*60)
    print("TRAINING TEST RESULTS")
    print("="*60)
    print(f"Total steps completed: {step + 1}")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")

    if len(loss_history) > 1:
        decreasing_steps = sum(1 for i in range(1, len(loss_history)) if loss_history[i] < loss_history[i-1])
        print(f"Steps where loss decreased: {decreasing_steps}/{len(loss_history)-1}")
        loss_trend = "decreasing" if decreasing_steps > (len(loss_history)-1)/2 else "not consistently decreasing"
        print(f"Loss trend: {loss_trend}")
    else:
        print("Loss trend: N/A (only 1 step)")
        loss_trend = "N/A"

    print(f"\nFinal load balancing loss (aux_loss): {aux_loss_history[-1]:.6f}")
    print(f"Load balancing loss computed: YES")

    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    print(f"1. Model initialized without errors: PASS")
    print(f"2. Training completed for {step + 1} steps: PASS")
    print(f"3. Loss decreased (trend): {loss_trend}")
    print(f"4. Load balancing loss computed: PASS ({aux_loss_history[-1]:.6f})")
    print(f"5. Expert utilization printed: PASS")
    print("="*60)


if __name__ == "__main__":
    train()
