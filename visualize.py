from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


TEAL   = "#1D9E75"
PURPLE = "#7F77DD"
CORAL  = "#D85A30"
GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})




class EpochRecord(NamedTuple):
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float   # 0–100



def plot_training_curves(history: list[EpochRecord], save_path: Path | None = None) -> None:
    """Dual-axis plot: loss curves (left) + val accuracy (right)."""
    epochs     = [r.epoch     for r in history]
    train_loss = [r.train_loss for r in history]
    val_loss   = [r.val_loss   for r in history]
    val_acc    = [r.val_acc    for r in history]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(epochs, train_loss, color=TEAL,   lw=1.5, marker="o", ms=4, label="train loss")
    ax1.plot(epochs, val_loss,   color=PURPLE, lw=1.5, marker="o", ms=4, label="val loss")
    ax2.plot(epochs, val_acc,    color=CORAL,  lw=1.5, marker="s", ms=4, linestyle="--", label="val acc %")

    ax1.set_xlabel("epoch");  ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy (%)", color=CORAL)
    ax2.tick_params(axis="y", colors=CORAL)
    ax2.spines["right"].set_edgecolor(CORAL)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#161b22", edgecolor="#30363d", fontsize=9, loc="center right")

    ax1.grid(True, axis="y")
    fig.suptitle("training curves", y=1.01, fontsize=11)
    fig.tight_layout()
    _show_or_save(fig, save_path, "training_curves.png")




def plot_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n: int = 32,
    save_path: Path | None = None,
) -> None:
    """imshow grid of test samples — green title = correct, red = wrong."""
    model.eval()
    images, labels, preds = [], [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred   = logits.argmax(dim=1)
            images.append(data.cpu())
            labels.append(target.cpu())
            preds.append(pred.cpu())
            if sum(len(b) for b in images) >= n:
                break

    images = torch.cat(images)[:n]
    labels = torch.cat(labels)[:n]
    preds  = torch.cat(preds)[:n]

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    axes = axes.flatten()

    imgs_np = (images.squeeze(1).numpy() * 0.3081 + 0.1307).clip(0, 1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs_np[i], cmap="gray", interpolation="nearest")
            correct = preds[i].item() == labels[i].item()
            color   = TEAL if correct else CORAL
            ax.set_title(f"p:{preds[i].item()}  t:{labels[i].item()}",
                         fontsize=7, color=color, pad=2)
        ax.axis("off")

    fig.suptitle(f"predictions  ({(preds == labels).float().mean()*100:.1f}% correct)",
                 fontsize=11, y=1.01)
    fig.tight_layout(pad=0.4)
    _show_or_save(fig, save_path, "predictions.png")


def plot_class_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of per-digit accuracy."""
    model.eval()
    correct = torch.zeros(10)
    total   = torch.zeros(10)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1).cpu()
            tgt  = target.cpu()
            for c in range(10):
                mask = tgt == c
                correct[c] += (pred[mask] == tgt[mask]).sum()
                total[c]   += mask.sum()

    acc = (correct / total * 100).numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(10), acc, color=TEAL, height=0.6, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8, color="#c9d1d9")
    ax.set_yticks(range(10)); ax.set_yticklabels([str(i) for i in range(10)])
    ax.set_xlabel("accuracy (%)"); ax.set_xlim(0, 105)
    ax.set_title("per-class accuracy", fontsize=11)
    ax.grid(True, axis="x")
    fig.tight_layout()
    _show_or_save(fig, save_path, "class_accuracy.png")



def plot_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path | None = None,
) -> None:
    """10×10 confusion matrix with counts and colour intensity."""
    model.eval()
    cm = torch.zeros(10, 10, dtype=torch.long)

    with torch.no_grad():
        for data, target in loader:
            pred = model(data.to(device)).argmax(dim=1).cpu()
            for t, p in zip(target, pred):
                cm[t.item(), p.item()] += 1

    cm_np = cm.numpy().astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_np, cmap="YlGn", interpolation="nearest")

    # annotate cells
    thresh = cm_np.max() / 2
    for r in range(10):
        for c in range(10):
            v = int(cm[r, c])
            ax.text(c, r, str(v), ha="center", va="center", fontsize=7,
                    color="black" if cm_np[r, c] > thresh else "#c9d1d9")

    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("confusion matrix", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _show_or_save(fig, save_path, "confusion_matrix.png")



def plot_conv_filters(model: torch.nn.Module, save_path: Path | None = None) -> None:
    """imshow grid of learned conv1 filters (32 × 3×3)."""
    weights = model.features[0].weight.detach().cpu()  # (32, 1, 3, 3)
    n = weights.shape[0]
    cols = 8; rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.0))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            w = weights[i, 0].numpy()
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax.imshow(w, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle("conv1 filters", fontsize=11, y=1.01)
    fig.tight_layout(pad=0.3)
    _show_or_save(fig, save_path, "conv_filters.png")



def plot_mistakes(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n: int = 24,
    save_path: Path | None = None,
) -> None:
    """imshow grid of the first N misclassified examples."""
    model.eval()
    wrong_imgs, wrong_true, wrong_pred = [], [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            mask = pred != target
            wrong_imgs.append(data[mask].cpu())
            wrong_true.append(target[mask].cpu())
            wrong_pred.append(pred[mask].cpu())
            if sum(len(b) for b in wrong_imgs) >= n:
                break

    imgs  = torch.cat(wrong_imgs)[:n]
    trues = torch.cat(wrong_true)[:n]
    preds = torch.cat(wrong_pred)[:n]
    imgs_np = (imgs.squeeze(1).numpy() * 0.3081 + 0.1307).clip(0, 1)

    cols = 8; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs_np[i], cmap="gray", interpolation="nearest")
            ax.set_title(f"p:{preds[i].item()} t:{trues[i].item()}",
                         fontsize=7, color=CORAL, pad=2)
        ax.axis("off")

    fig.suptitle("misclassified examples", fontsize=11, y=1.01)
    fig.tight_layout(pad=0.4)
    _show_or_save(fig, save_path, "mistakes.png")


def plot_dashboard(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    history: list[EpochRecord],
    save_path: Path | None = None,
) -> None:
    """Single-figure dashboard: curves + predictions + class acc + conf matrix."""
    model.eval()


    images, labels, preds_list = [], [], []
    cm = torch.zeros(10, 10, dtype=torch.long)
    correct_cls = torch.zeros(10); total_cls = torch.zeros(10)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            images.append(data.cpu()); labels.append(target.cpu()); preds_list.append(pred.cpu())
            for t, p in zip(target.cpu(), pred.cpu()):
                cm[t.item(), p.item()] += 1
            p_cpu, t_cpu = pred.cpu(), target.cpu()
            for c in range(10):
                mask = t_cpu == c
                correct_cls[c] += (p_cpu[mask] == t_cpu[mask]).sum()
                total_cls[c]   += mask.sum()

    images = torch.cat(images)[:16]
    labels = torch.cat(labels)[:16]
    preds_t = torch.cat(preds_list)[:16]
    imgs_np = (images.squeeze(1).numpy() * 0.3081 + 0.1307).clip(0, 1)
    acc_cls = (correct_cls / total_cls * 100).numpy()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    gs_imgs = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=gs[0, :], hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(gs_imgs[i // 8, i % 8])
        ax.imshow(imgs_np[i], cmap="gray", interpolation="nearest")
        correct = preds_t[i].item() == labels[i].item()
        ax.set_title(f"p:{preds_t[i].item()} t:{labels[i].item()}",
                     fontsize=6.5, color=TEAL if correct else CORAL, pad=1.5)
        ax.axis("off")

    ax_loss = fig.add_subplot(gs[1, 0])
    epochs     = [r.epoch for r in history]
    ax_loss.plot(epochs, [r.train_loss for r in history], color=TEAL,   lw=1.5, marker="o", ms=3, label="train")
    ax_loss.plot(epochs, [r.val_loss   for r in history], color=PURPLE, lw=1.5, marker="o", ms=3, label="val")
    ax_loss.set_title("loss", fontsize=10); ax_loss.set_xlabel("epoch"); ax_loss.grid(True)
    ax_loss.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")


    ax_acc = fig.add_subplot(gs[1, 1])
    ax_acc.barh(range(10), acc_cls, color=TEAL, height=0.6, alpha=0.85)
    ax_acc.set_yticks(range(10)); ax_acc.set_yticklabels(range(10))
    ax_acc.set_xlim(0, 105); ax_acc.set_title("per-class accuracy", fontsize=10)
    ax_acc.set_xlabel("accuracy (%)"); ax_acc.grid(True, axis="x")

    ax_cm = fig.add_subplot(gs[1, 2])
    cm_np = cm.numpy().astype(float)
    im = ax_cm.imshow(cm_np, cmap="YlGn", interpolation="nearest")
    thresh = cm_np.max() / 2
    for r in range(10):
        for c in range(10):
            ax_cm.text(c, r, str(int(cm[r, c])), ha="center", va="center", fontsize=5,
                       color="black" if cm_np[r, c] > thresh else "#c9d1d9")
    ax_cm.set_xticks(range(10)); ax_cm.set_yticks(range(10))
    ax_cm.set_xlabel("predicted"); ax_cm.set_ylabel("true")
    ax_cm.set_title("confusion matrix", fontsize=10)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    fig.suptitle("MNIST CNN — evaluation dashboard", fontsize=13, y=1.01)
    _show_or_save(fig, save_path, "dashboard.png")



def _show_or_save(fig: plt.Figure, save_path: Path | None, default_name: str) -> None:
    if save_path:
        out = Path(save_path)
        out.mkdir(parents=True, exist_ok=True)
        p = out / default_name
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info("Saved → %s", p)
        plt.close(fig)
    else:
        plt.show()