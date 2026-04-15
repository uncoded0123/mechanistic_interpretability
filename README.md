# Mechanistic Interpretability

Hands-on mechanistic interpretability experiments.

## Projects

### MNIST From Scratch — [`MNIST From Scratch.ipynb`](MNIST%20From%20Scratch.ipynb)

2-hidden-layer neural network with manual forward and backward pass. No `loss.backward()`, no `optim.step()`. 97.94% accuracy in under 2 minutes on CPU.

### Raw GPT-2 + Steering — [`gpt2/raw_gpt2_with_mech_interp.ipynb`](gpt2/raw_gpt2_with_mech_interp.ipynb)

GPT-2 small forward pass from raw weight tensors — attention, MLP, and layer norm implemented manually. Includes difference-in-means steering to shift output toward Golden Gate Bridge / San Francisco concepts.

No steer: "the beach. It was a beautiful place..."

Steered:  "the San Francisco Bay Bridge in the early 1970s..."

### Minimal Transformer — [`minimal_transformer/`](minimal_transformer/)

Character-level transformer trained from scratch on Shakespeare. Includes attention heatmaps, SAE training, and steering.

## Setup

```bash
pip install transformers torch torchvision tiktoken
```

MNIST runs on CPU (8GB+ RAM). Raw GPT-2 and minimal transformer notebooks run faster on GPU (Colab T4 works).
