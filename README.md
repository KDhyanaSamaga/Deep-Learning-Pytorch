<div align="center">

<img src="https://pytorch.org/assets/images/pytorch-logo.png" width="120" alt="PyTorch Logo" />

# PyTorch Explained
### An Engineer-Friendly Deep Dive

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Medium](https://img.shields.io/badge/Read%20on-Medium-000000?style=flat-square&logo=medium&logoColor=white)](https://medium.com/@kdhyanasamaga)

*A structured walkthrough of PyTorch fundamentals — from tensors to transformers, designed for students, engineers, and researchers.*

</div>

---

## 📖 What's Covered

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Introduction](#1-introduction-to-pytorch) | What PyTorch is and why it matters |
| 2 | [Torch vs PyTorch](#2-torch-vs-pytorch) | History and key differences |
| 3 | [Dynamic Computation Graph](#3-dynamic-computation-graph) | Define-by-Run explained |
| 4 | [Tensors & Core Modules](#4-core-components-and-tensors) | The building blocks of PyTorch |
| 5 | [Autograd](#5-autograd) | Automatic differentiation under the hood |
| 6 | [Data Pipelines](#6-data-pipelines) | Dataset and DataLoader patterns |
| 7 | [Optimization & Regularization](#7-optimization-and-regularization) | Fighting overfitting |
| 8 | [CNNs & Transfer Learning](#8-cnns-and-transfer-learning) | Vision models and fine-tuning |
| 9 | [RNNs](#9-recurrent-neural-networks) | Sequential data modeling |

---

## 1. Introduction to PyTorch

PyTorch is an **open-source deep learning framework developed by Meta (Facebook)**. It has become the dominant framework in AI research and is widely adopted in production systems.

**Built for:**
- Deep Learning & Neural Networks
- Computer Vision
- Natural Language Processing
- AI Research & Rapid Prototyping

**Why engineers prefer it:**
- Pythonic and intuitive API
- Dynamic graphs = easier debugging
- Native GPU acceleration
- Massive community and ecosystem

---

## 2. Torch vs PyTorch

PyTorch evolved from the older **Torch** framework.

| Feature | Torch | PyTorch |
|--------|-------|---------|
| Language | Lua | Python |
| Graph type | Static | Dynamic |
| Debugging | Hard | Easy |
| Ecosystem | Limited | Rich |
| Adoption | Low | Very High |

> Torch used **Static Computation Graphs**, requiring the full graph to be defined before execution. PyTorch replaced this with the far more flexible **Define-by-Run** approach.

---

## 3. Dynamic Computation Graph

PyTorch builds the computation graph **on the fly during execution**, not upfront.

```
Forward Pass:
  input → [op1] → [op2] → [op3] → output
                                      ↓
                                   loss
Backward Pass:
  input ← [grad1] ← [grad2] ← [grad3] ← loss.backward()
```

**Why this matters:**
- Supports dynamic control flow (`if`, `for`, `while` in model logic)
- Natural Python debugging with `print()` and breakpoints
- Faster research iterations — change architecture mid-run
- Essential for variable-length inputs (NLP, graphs)

---

## 4. Core Components and Tensors

### Tensors

Tensors are PyTorch's **primary data structure** — like NumPy arrays but with GPU support and autograd integration.

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x = x.to("cuda")           # Move to GPU
x = torch.randn(3, 224, 224)  # Random tensor (C, H, W)
```

**Tensor dimensions and use cases:**

| Rank | Shape Example | Common Use |
|------|---------------|------------|
| 1D | `(512,)` | Word embeddings, feature vectors |
| 2D | `(32, 512)` | Batch of vectors, weight matrices |
| 3D | `(3, 224, 224)` | Single RGB image (C, H, W) |
| 4D | `(32, 3, 224, 224)` | Batch of images (N, C, H, W) |
| 5D | `(8, 16, 3, 112, 112)` | Video batches (N, T, C, H, W) |

---

### Essential Modules

#### `torch` — Tensor computation engine
```python
import torch
t = torch.randn(4, 4)
t.shape     # torch.Size([4, 4])
t.dtype     # torch.float32
```

#### `torch.nn` — Neural network building blocks
```python
import torch.nn as nn

layer = nn.Linear(128, 64)
relu  = nn.ReLU()
conv  = nn.Conv2d(3, 64, kernel_size=3, padding=1)
loss  = nn.CrossEntropyLoss()
```

#### `torch.optim` — Optimization algorithms
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### `torch.autograd` — Automatic differentiation
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)  # 12.0 → dy/dx = 3x²
```

#### `torch.jit` — Serialization & deployment
```python
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

---

## 5. Autograd

Autograd is PyTorch's **automatic differentiation engine**. It tracks every operation on tensors with `requires_grad=True` and computes gradients automatically during backpropagation.

### Training Loop

```python
for epoch in range(num_epochs):
    # 1. Forward pass
    outputs = model(inputs)

    # 2. Compute loss
    loss = criterion(outputs, labels)

    # 3. Zero gradients (prevent accumulation)
    optimizer.zero_grad()

    # 4. Backward pass (compute gradients)
    loss.backward()

    # 5. Update weights
    optimizer.step()
```

### Gradient Flow

```
loss.backward()
      ↓
  Traverses computation graph in reverse
      ↓
  Populates .grad for all leaf tensors
      ↓
  optimizer.step() uses those gradients
```

> **Tip:** Always call `optimizer.zero_grad()` before `loss.backward()`. Gradients accumulate by default — a common source of subtle bugs.

---

## 6. Data Pipelines

PyTorch provides `Dataset` and `DataLoader` for efficient, scalable data loading.

### Custom Dataset

```python
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = load_image(self.file_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
```

### DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Parallel loading
    pin_memory=True     # Faster GPU transfer
)

for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)
    # training step...
```

---

## 7. Optimization and Regularization

Deep models overfit without regularization. PyTorch includes first-class support for all standard techniques.

### Dropout
```python
self.dropout = nn.Dropout(p=0.3)
# Randomly zeroes 30% of neurons during training
```

### Batch Normalization
```python
self.bn = nn.BatchNorm2d(num_features=64)
# Normalizes activations across the batch
```

### L2 Regularization (Weight Decay)
```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Penalizes large weights during optimization
```

### Learning Rate Scheduler
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Decays LR by 10x every 10 epochs
```

### Early Stopping (Manual)
```python
best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 5

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break
```

---

## 8. CNNs and Transfer Learning

### Convolutional Neural Networks

CNNs are the backbone of modern computer vision. They learn spatial hierarchies of features.

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**CNN layer roles:**
- `Conv2d` → Feature extraction (edges, textures, patterns)
- `BatchNorm2d` → Stabilize training
- `MaxPool2d` → Spatial downsampling
- `Linear` → Classification head

---

### Transfer Learning

Why train from scratch when pretrained models exist?

```python
import torchvision.models as models

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

# Only the new layer trains
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

**Available pretrained models:**

| Model | Params | Top-1 Acc (ImageNet) | Best For |
|-------|--------|----------------------|---------|
| ResNet50 | 25M | 76.1% | General purpose |
| EfficientNet-B4 | 19M | 83.0% | Efficiency-accuracy tradeoff |
| ViT-B/16 | 86M | 81.8% | Large datasets |
| MobileNetV3 | 5M | 75.2% | Edge/mobile deployment |

---

## 9. Recurrent Neural Networks

RNNs process **sequential data** by maintaining a hidden state across time steps.

```
Input sequence:  x₁ → x₂ → x₃ → x₄
                  ↓     ↓     ↓     ↓
Hidden state:   h₁ → h₂ → h₃ → h₄ → output
```

### RNN Variants in PyTorch

```python
# Vanilla RNN (rarely used — vanishing gradient problem)
rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)

# LSTM — handles long-range dependencies
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               batch_first=True, dropout=0.3)

# GRU — lighter alternative to LSTM
gru = nn.GRU(input_size=128, hidden_size=256, batch_first=True)
```

### Example: Text Classification with LSTM

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # (B, T) → (B, T, E)
        out, (hn, cn) = self.lstm(x)       # hn: final hidden state
        return self.fc(hn[-1])             # Classify from last hidden state
```

**RNN Applications:**

| Task | Architecture | Example |
|------|-------------|---------|
| Sentiment Analysis | LSTM | Movie review → Positive/Negative |
| Language Modeling | LSTM/GRU | Predict next word |
| Time Series | GRU | Stock price forecasting |
| Sequence Labeling | Bi-LSTM | Named Entity Recognition |
| Speech Recognition | LSTM + CTC | Audio → Text |

---

## Conclusion

PyTorch gives you the tools to go from idea to working model quickly, without sacrificing the flexibility needed for research or the performance needed for production.

**What you can build with PyTorch:**
- Image classifiers, object detectors, segmentation models
- Chatbots, translation systems, sentiment analyzers
- Time-series forecasters, anomaly detectors
- Generative models (GANs, VAEs, Diffusion)
- Custom research architectures

The key to mastering PyTorch is **building projects**. Read the docs, break things, fix them.

---

## Author

<div align="center">

**K Dhyana Samaga**

B.E. in Artificial Intelligence & Machine Learning  
Srinivas Institute of Technology, Mangaluru

[![Medium](https://img.shields.io/badge/Medium-@kdhyanasamaga-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@kdhyanasamaga)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

*Interested in Computer Vision · Deep Learning · Backend Engineering · AI Systems*

</div>

---

<div align="center">
<sub>If this helped you, consider leaving a ⭐ on the repo.</sub>
</div>
