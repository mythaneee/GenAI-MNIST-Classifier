# MNIST Handwritten Digit Classification

A convolutional neural network (CNN) built with PyTorch to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), with a rich suite of visualizations.

---

## Project Structure

```
├── mnist_classifier.py   # Model definition, training loop, and entry point
├── visualize.py          # All plotting utilities
├── data/
│   └── MNIST/
│       └── raw/          # Raw MNIST binary files (auto-downloaded if missing)
└── README.md
```

---

## Model Architecture

| Layer | Details |
|---|---|
| Conv1 | 1 → 32 filters, 3×3, padding=1, ReLU |
| Conv2 | 32 → 32 filters, 3×3, padding=1, ReLU |
| MaxPool | 2×2 |
| Dropout | p=0.25 |
| Flatten | 32 × 14 × 14 = 6272 |
| FC1 | 6272 → 128, ReLU |
| Dropout | p=0.50 |
| FC2 | 128 → 64, ReLU |
| FC3 | 64 → 10, log-softmax |

Loss function: **Negative Log-Likelihood (NLL)**  
Optimizer: **Adam**

---

## Hyperparameters

| Parameter | Default |
|---|---|
| `BATCH_SIZE` | 64 |
| `LR` | 1e-3 |
| `EPOCH` | 2 |
| `VIS_SAVE_DIR` | `None` (display inline) |

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

---

## Usage

```bash
python mnist_classifier.py
```

The script will:
1. Download MNIST into `./data/` automatically if not already present.
2. Train the CNN for `EPOCH` epochs, logging progress every 100 batches.
3. Evaluate on the test set and report average loss and accuracy.
4. Display the following visualizations (or save them if `VIS_SAVE_DIR` is set).

### Saving Plots

Set `VIS_SAVE_DIR` in `mnist_classifier.py` to a directory path to save all plots as PNG files instead of displaying them interactively:

```python
VIS_SAVE_DIR = "./outputs"
```

---

## Visualizations

| Plot | Description |
|---|---|
| Training Curves | Dual-axis: train/val loss (left) + val accuracy % (right) |
| Predictions | Image grid with green (correct) / red (incorrect) titles |
| Class Accuracy | Per-digit accuracy bar chart |
| Confusion Matrix | 10×10 heatmap of predicted vs. true labels |
| Mistakes | Grid of misclassified samples with true and predicted labels |
| Dashboard | Combined summary figure of all key metrics |

---

## Dataset

The MNIST dataset consists of 70,000 grayscale 28×28 images of handwritten digits (0–9):
- **Training set**: 60,000 images
- **Test set**: 10,000 images

Normalization applied: mean=`0.1307`, std=`0.3081`.
