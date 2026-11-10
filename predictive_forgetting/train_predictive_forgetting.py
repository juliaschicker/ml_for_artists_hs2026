#!/usr/bin/env python3
"""
Phase 1 training: learn the generative model (encoder + decoder) that a
later wake/sleep consolidation script will dream through.

Directly follows the paper's Methods, "Cortical Refinement (Autoencoder)
Model" (Section B.3, Phase 1): a convolutional autoencoder trained with
BCE-with-logits reconstruction loss, AdamW, gradient clipping, and early
stopping on a held-out validation split. Same hyperparameters as the
paper's own Phase 1 recipe (lr 1e-3, wd 1e-4, grad clip 5.0, patience 5).

On top of plain reconstruction, each frame x_t is also paired with a
successor target y_t -- per the paper's Section 2.2 definition of Y for
unsupervised/observational settings ("typically the successor state
X_{t+1}") -- and the encoder is trained to make z_t predictive of y_t
via a small successor-predictor head sharing the same decoder. y_t is
the *mean* of a short window of future frames rather than a single
frame: a lone future frame is nearly identical to x_t at native frame
rate (a trivial, unselective target) and is itself noisy; averaging a
window gives a real prediction gap and cancels per-frame noise out of
the target. This is the mechanism that makes forgetting *predictive*
rather than merely an artifact of a small bottleneck (see discussion
in conversation: a restricted-dataset autoencoder alone compresses
toward what's statistically common, not toward what's predictive).

This script does NOT implement predictive forgetting's compression
step itself -- it only trains Dec_Theta (and h_Psi) to be a competent,
future-aware generative model of the source footage. Consolidation
(wake/sleep, free-energy inference) happens in a separate script, at
inference time, using the frozen model trained here.

Usage:
    python train_predictive_forgetting.py
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from predictive_forgetting_model import PredictiveForgettingAutoencoder

# ─── PARAMETERS ─────────────────────────────────────────────────────────────

FRAMES_DIR     = "training_footage_260917_choreo_frames"  # directory of source frames (any cv2-readable image)
FRAME_GLOB     = "*.jpg"

CHECKPOINT_OUT = "predictive_forgetting_model_succ_14_260917_256_144.pt"

# (height, width) frames are resized to -- 16:9, both dims divisible by
# 2^len(CHANNELS)=16 (required for the 4 stride-2 downsampling layers below).
INPUT_SIZE     = (144, 256)
CHANNELS       = [32, 64, 128, 128]  # encoder/decoder conv channel progression
LATENT_DIM     = 96

# Successor target y_t = mean(frames[t+1 : t+1+SUCCESSOR_WINDOW]).
# At 25fps: 8 frames ~ 0.32s ahead. Widen this if your footage moves slowly
# (the target should differ meaningfully from x_t -- if it barely does,
# the successor objective has nothing to select against).
SUCCESSOR_WINDOW      = 28          # make this as large as a multiple of a beat (e.g. 1, 2 or 4)
SUCCESSOR_LOSS_WEIGHT = 0.5         # experimental knob -- retune once looking at real output

VAL_FRACTION   = 0.10              # held-out validation split, per paper's Methods
BATCH_SIZE     = 16
EPOCHS         = 20
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 5.0
PATIENCE       = 5                 # early stopping patience, per paper's Methods

SEED           = 0

# ────────────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FrameDataset(Dataset):
    """
    Yields (x_t, y_t) pairs, both [0, 1] float tensors resized to
    (height, width) = input_size (may be non-square, e.g. 16:9 frames):
      x_t = frame at index t
      y_t = mean of frames [t+1, t+1+SUCCESSOR_WINDOW) -- the successor
            state Y (paper Section 2.2), averaged over a short future
            window to give a real prediction gap and cancel per-frame noise.

    All frames are decoded and resized once up front and cached in memory
    (tens of MB at this resolution). Without this, each frame would be
    re-read from disk and re-decoded up to SUCCESSOR_WINDOW+1 times per
    epoch, since consecutive samples' successor windows overlap heavily.
    """

    def __init__(self, frame_paths: list, input_size: tuple, successor_window: int):
        self.successor_window = successor_window
        print(f"  caching {len(frame_paths)} frames in memory...")
        self.frames = torch.stack([self._load(p, input_size) for p in frame_paths])
        self.valid_t = list(range(len(frame_paths) - successor_window))

    @staticmethod
    def _load(path, input_size: tuple) -> torch.Tensor:
        height, width = input_size
        img = cv2.imread(str(path))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)  # cv2 takes (w, h)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = self.valid_t[idx]
        x = self.frames[t]
        y = self.frames[t + 1: t + 1 + self.successor_window].mean(dim=0)
        return x, y


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> tuple:
    train = optimizer is not None
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()
    total_recon, total_succ, total_n = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            z = model.encode(x)
            recon_loss = criterion(model.decode(z), x)
            succ_loss = criterion(model.predict_successor(z), y)
            loss = recon_loss + SUCCESSOR_LOSS_WEIGHT * succ_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_recon += recon_loss.item() * x.size(0)
            total_succ += succ_loss.item() * x.size(0)
            total_n += x.size(0)

    return total_recon / total_n, total_succ / total_n


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    base = Path(__file__).parent
    frame_paths = sorted((base / FRAMES_DIR).glob(FRAME_GLOB))
    if not frame_paths:
        print(f"No frames found matching '{FRAME_GLOB}' in {base / FRAMES_DIR}")
        return

    device = get_device()
    print("Predictive forgetting -- Phase 1 autoencoder training")
    print(f"  {len(frame_paths)} frames · device={device}")

    dataset = FrameDataset(frame_paths, INPUT_SIZE, SUCCESSOR_WINDOW)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    if n_train < 1:
        print(f"Not enough frames ({len(dataset)}) for a train/val split "
              f"(need more than SUCCESSOR_WINDOW={SUCCESSOR_WINDOW}).")
        return

    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"  train={n_train} · val={n_val}\n")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = PredictiveForgettingAutoencoder(
        in_channels=3, channels=CHANNELS, latent_dim=LATENT_DIM, input_size=INPUT_SIZE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_recon, train_succ = run_epoch(model, train_loader, device, optimizer)
        val_recon, val_succ = run_epoch(model, val_loader, device, optimizer=None)
        train_loss = train_recon + SUCCESSOR_LOSS_WEIGHT * train_succ
        val_loss = val_recon + SUCCESSOR_LOSS_WEIGHT * val_succ
        print(f"  epoch {epoch:02d}/{EPOCHS}  "
              f"train={train_loss:.4f} (recon={train_recon:.4f} succ={train_succ:.4f})  "
              f"val={val_loss:.4f} (recon={val_recon:.4f} succ={val_succ:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_channels": 3,
                "channels": CHANNELS,
                "latent_dim": LATENT_DIM,
                "input_size": INPUT_SIZE,
                "successor_window": SUCCESSOR_WINDOW,
                "successor_loss_weight": SUCCESSOR_LOSS_WEIGHT,
                "val_loss": val_loss,
            }, base / CHECKPOINT_OUT)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\n  early stopping (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
