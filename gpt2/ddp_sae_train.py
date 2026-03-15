# Multi-GPU SAE Training with DDP
# Run: python -m torch.distributed.launch --nproc_per_node=2 ddp_sae_train.py
# Or:  torchrun --nproc_per_node=2 ddp_sae_train.py
#
# DDP = same training loop + 3 lines:
#   1. dist.init_process_group()  — connect GPUs
#   2. DDP(model)                 — auto-sync gradients
#   3. DistributedSampler()       — split data across GPUs
#
# Each GPU runs this SAME script with different rank.
# mp.spawn launches N parallel processes (not a loop).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import tiktoken
from transformers import GPT2Model
import os
import time

# ==================== GPT-2 Raw Forward ====================
# Needed to extract activations for SAE training

def load_gpt2():
    """Load GPT-2 weights as flat dict of tensors"""
    sd = GPT2Model.from_pretrained("gpt2").state_dict()
    enc = tiktoken.get_encoding("gpt2")
    return sd, enc

def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias

def gelu(x):
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

def raw_forward(tokens, sd, sae_layer=8):
    """Forward pass through GPT-2, return residual stream at sae_layer"""
    seq_len = tokens.shape[1]
    n_heads, d_head = 12, 64

    emb = sd['wte.weight'][tokens[0]]
    pos = sd['wpe.weight'][:seq_len]
    x = (emb + pos).unsqueeze(0)

    for layer in range(sae_layer):
        # Attention
        ln1_out = layer_norm(x, sd[f'h.{layer}.ln_1.weight'], sd[f'h.{layer}.ln_1.bias'])
        W_qkv = sd[f'h.{layer}.attn.c_attn.weight']
        b_qkv = sd[f'h.{layer}.attn.c_attn.bias']
        q = ln1_out @ W_qkv[:, :768] + b_qkv[:768]
        k = ln1_out @ W_qkv[:, 768:1536] + b_qkv[768:1536]
        v = ln1_out @ W_qkv[:, 1536:] + b_qkv[1536:]
        q = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
        k = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)
        v = v.view(1, seq_len, n_heads, d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        pattern = torch.softmax(scores, dim=-1)
        z = (pattern @ v).transpose(1, 2).contiguous().view(1, seq_len, 768)
        attn_out = z @ sd[f'h.{layer}.attn.c_proj.weight'] + sd[f'h.{layer}.attn.c_proj.bias']
        x = x + attn_out

        # MLP
        ln2_out = layer_norm(x, sd[f'h.{layer}.ln_2.weight'], sd[f'h.{layer}.ln_2.bias'])
        hidden = gelu(ln2_out @ sd[f'h.{layer}.mlp.c_fc.weight'] + sd[f'h.{layer}.mlp.c_fc.bias'])
        mlp_out = hidden @ sd[f'h.{layer}.mlp.c_proj.weight'] + sd[f'h.{layer}.mlp.c_proj.bias']
        x = x + mlp_out

    return x  # [1, seq, 768]


# ==================== Collect Activations ====================

def collect_activations(sd, enc):
    """Run GPT-2 on prompts, collect residual stream vectors at layer 8"""
    prompts = [
        "The Golden Gate Bridge spans the San Francisco Bay connecting the city to Marin County",
        "Machine learning models learn patterns from data by adjusting weights through backpropagation",
        "The capital of France is Paris which is known for the Eiffel Tower and the Louvre Museum",
        "Scientists discovered that the universe is expanding at an accelerating rate due to dark energy",
        "Python is a popular programming language used for web development and artificial intelligence",
        "The history of ancient Rome spans over a thousand years from kingdom to republic to empire",
        "Quantum computers use qubits that can exist in superposition of states unlike classical bits",
        "The Amazon rainforest produces roughly twenty percent of the world oxygen supply each year",
        "Neural networks consist of layers of neurons connected by weights that are learned during training",
        "Shakespeare wrote thirty seven plays including Hamlet Macbeth and Romeo and Juliet among others",
        "The stock market crashed in nineteen twenty nine leading to the Great Depression across the world",
        "Photosynthesis converts sunlight water and carbon dioxide into glucose and oxygen in plant cells",
        "The internet was originally developed by DARPA as a military communication network in the sixties",
        "Einstein published his theory of general relativity in nineteen fifteen changing physics forever",
        "DNA contains the genetic instructions for the development and functioning of all living organisms",
        "The moon orbits the Earth approximately every twenty seven days and causes ocean tides on Earth",
        "Artificial intelligence has made significant progress in natural language processing and computer vision",
        "The Great Wall of China stretches over thirteen thousand miles across northern China border regions",
        "Economists study how societies allocate scarce resources among competing wants and unlimited needs",
        "The human brain contains approximately eighty six billion neurons connected by trillions of synapses",
    ]

    all_acts = []
    with torch.no_grad():
        for p in prompts:
            token_ids = [50256] + enc.encode(p)
            tokens = torch.tensor([token_ids])
            resid = raw_forward(tokens, sd)
            all_acts.append(resid[0].clone())

    return torch.cat(all_acts, dim=0)  # [total_tokens, 768]


# ==================== SAE Model ====================

class SAE(nn.Module):
    """SAE = 1 layer (expand) MSE neural net with L1"""
    def __init__(self, d_model=768, d_sae=4096):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # Init decoder to unit norm rows
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1)

    def forward(self, x):
        h = torch.relu(self.encoder(x - self.b_dec))  # [batch, 4096]
        x_hat = self.decoder(h) + self.b_dec           # [batch, 768]
        recon_loss = (x - x_hat).pow(2).mean()
        l1_loss = h.abs().mean()
        return x_hat, recon_loss, l1_loss, h


# ==================== DDP Training ====================

def train_sae_distributed(rank, world_size, activations):
    """
    rank       = this GPU's ID (0 or 1)
    world_size = total GPUs
    activations = [N, 768] tensor of residual stream vectors
    """
    # --- DDP line 1: connect GPUs ---
    os.environ["MASTER_ADDR"] = "localhost"  # all GPUs meet on this computer
    os.environ["MASTER_PORT"] = "12355"      # at this port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    # --- DDP line 2: wrap model ---
    model = SAE(d_model=768, d_sae=4096).to(device)
    model = DDP(model, device_ids=[rank])

    # --- DDP line 3: split data ---
    dataset = TensorDataset(activations)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    l1_coeff = 3e-3

    # --- Training loop (same as single-GPU) ---
    start = time.time()
    for epoch in range(20):
        sampler.set_epoch(epoch)  # shuffle per epoch (deterministic per rank)
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, recon_loss, l1_loss, h = model(batch)
            loss = recon_loss + l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()  # DDP averages gradients HERE automatically
            optimizer.step()

            # Normalize decoder to unit norm
            with torch.no_grad():
                model.module.decoder.weight.data = F.normalize(
                    model.module.decoder.weight.data, dim=1
                )

        if rank == 0:
            frac_active = (h > 0).float().mean().item()
            print(f"Epoch {epoch:>3} | loss={loss.item():.4f} | recon={recon_loss.item():.4f} | "
                  f"l1={l1_loss.item():.4f} | active={frac_active:.1%}")

    if rank == 0:
        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s on {world_size} GPUs")
        # Save model
        torch.save(model.module.state_dict(), "sae_ddp_trained.pt")
        print("Saved: sae_ddp_trained.pt")

    dist.destroy_process_group()


# ==================== Main ====================

if __name__ == "__main__":
    print("Loading GPT-2 and collecting activations...")
    sd, enc = load_gpt2()
    activations = collect_activations(sd, enc)
    print(f"Collected {activations.shape[0]} activation vectors")

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")

    if world_size < 2:
        print("Need 2+ GPUs for DDP. Exiting.")
        exit(1)

    print(f"Launching DDP training on {world_size} GPUs...")
    mp.spawn(train_sae_distributed, args=(world_size, activations), nprocs=world_size)
