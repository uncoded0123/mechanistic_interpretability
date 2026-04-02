# Mechanistic Interpretability

Hands-on mechanistic interpretability experiments.

## Projects

### MNIST From Scratch — [`MNIST From Scratch.ipynb`](MNIST%20From%20Scratch.ipynb)

2-hidden-layer neural network with manual forward and backward pass. No `loss.backward()`, no `optim.step()`. 97.94% accuracy in under 2 minutes on CPU.

### Raw GPT-2 + Steering — [`gpt2/raw_gpt2_with_mech_interp.ipynb`](gpt2/raw_gpt2_with_mech_interp.ipynb)

GPT-2 small forward pass from raw weight tensors — attention, MLP, and layer norm implemented manually. Includes difference-in-means steering to shift output toward Golden Gate Bridge / San Francisco concepts.

### Pretrained SAE Steering — [`gpt2/gpt2_mech_interp.ipynb`](gpt2/gpt2_mech_interp.ipynb)

Golden Gate Bridge steering using a pretrained sparse autoencoder (24,576 features) via TransformerLens + sae_lens. Identifies feature 23937 and injects its decoder direction into GPT-2's residual stream.

### Minimal Transformer — [`minimal_transformer/`](minimal_transformer/)

Character-level transformer trained from scratch on Shakespeare. Includes attention heatmaps, SAE training, and steering.

## Setup

```bash
pip install transformer_lens sae_lens transformers torch torchvision tiktoken
```

MNIST, raw GPT-2, and minimal transformer run on CPU (8GB+ RAM). Pretrained SAE steering and heatmap notebooks require GPU (Colab T4 works).
