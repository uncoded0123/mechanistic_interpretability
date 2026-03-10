# Mechanistic Interpretability

Hands-on mechanistic interpretability experiments on GPT-2 small (124M params). All library abstractions stripped down to raw tensor operations.

## Projects

### GPT-2 SAE Circuit Tracing — [`gpt2/gpt2_mech_interp.ipynb`](gpt2/gpt2_mech_interp.ipynb)

Sparse autoencoder analysis and circuit tracing on GPT-2 small, running locally on CPU.

**What's in it:**
- Load SAE as 4 raw tensors (W_enc, b_enc, W_dec, b_dec) — no library encode/decode calls
- SAE encode from first principles: `features = ReLU((x - b_dec) @ W_enc + b_enc)`
- Feature identification: find which SAE features correspond to a concept (e.g. "Golden Gate Bridge")
- Feature verification: probe what concept a given feature represents across diverse prompts
- Steering: add scaled SAE decoder directions to GPT-2's residual stream to shift model output
- **Circuit tracing via ablation**: zero out individual attention heads and MLP layers, measure downstream effect on SAE feature activation

**Key findings:**

**1. Unigram circuit (Cell 10)** — SAE feature 10261 (" Golden") is 85% driven by MLP layer 0. No attention heads contribute. It's a token-identity feature computed directly from the embedding.

| Component | Δ SAE activation | Role |
|-----------|-----------------|------|
| L0 MLP | -62.97 | Primary driver |
| L7 MLP | -14.90 | Secondary refinement |
| All attention heads | 0.00 | No contribution |

**2. Contextual circuit (Cell 11)** — IOI task: "John and Mary went to the store. John gave a drink to" → model predicts "Mary". Ablation reveals attention heads that move name information across positions.

| Component | Δ logit("Mary") | Role |
|-----------|----------------|------|
| L0 MLP | -4.30 | Token processing |
| L10 MLP | -2.19 | Late refinement |
| L11 Attn H0 | -1.90 | Name mover head |
| L0 Attn H11 | +1.57 | Negative/inhibition head |
| L11 Attn H10 | +1.39 | Backup negative head |
| L5 Attn H9 | -0.96 | S-inhibition head |

The contrast shows two types of circuits: MLP-only (unigram features driven by token identity) vs attention-mediated (contextual features requiring cross-position information flow).

### Minimal Transformer — [`minimal_transformer/minimal_transformer.ipynb`](minimal_transformer/minimal_transformer.ipynb)

Character-level transformer trained from scratch on Shakespeare. Includes custom SAE training and steering.

## Setup

```bash
pip install transformer_lens sae_lens transformers torch
```

Runs on CPU (8GB+ RAM). No GPU required for inference.