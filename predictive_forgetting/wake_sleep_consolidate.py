#!/usr/bin/env python3
"""
Wake/sleep consolidation ("dreaming") -- the predictive forgetting step
itself.

Directly implements the paper's free-energy formulation and Langevin
inference dynamics (Methods, "Predictive Coding Networks", Eq. 16-17):

    F(x, z) = ||Dec_Theta(z) - x||^2 + beta * ||z||^2
    z <- z - eta * grad_z F(x, z) + sigma * N(0, I)

Wake: settle z against the REAL frame x under a WEAK prior (low beta) --
high-fidelity, matches the paper's wake regime.

Sleep (one consolidation cycle): decode the current z into a dream
x_hat = sigmoid(Dec_Theta(z)); re-settle z against that dream (not the
original frame) under a STRONG prior (high beta) -- this is the actual
compression step, forcing z toward whatever Dec_Theta can confidently
regenerate rather than the specific input. Repeating this cycle N times
is N "sleep cycles"; per the paper's Data-Processing-Inequality argument
(Eq. 5), each cycle can only lose information relative to the last, so
the frame progressively consolidates toward the model's own attractor
rather than merely accumulating blur.

Requires a checkpoint from train_predictive_forgetting.py. Dec_Theta's
(and Enc_Phi's) weights are frozen throughout -- only z is ever updated
here. The successor predictor from training is not used in this script;
it only shaped the encoder during Phase 1.

Usage:
    python 03_wake_sleep_consolidate.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from predictive_forgetting_model import PredictiveForgettingAutoencoder

# ─── PARAMETERS ─────────────────────────────────────────────────────────────

CHECKPOINT   = "predictive_forgetting_model_re_latent_dim_16_succ_3.pt"

FRAMES_DIR   = "batch_A"
FRAME_GLOB   = "*.jpg"
OUTPUT_DIR   = "consolidated_predictive_forgetting_re_latent_dim_16_succ_3"

BATCH_SIZE   = 16     # frames are consolidated independently, batched for throughput
TEST_FRAMES  = 0      # >0 to process only a slice of N frames (quick test)
TEST_FRAME_START = 0  # slice starts here -- e.g. set to a frame index where
                       # something is actually happening in the footage

# TODO play around with sleep cycles, the more cycles, the more the model will forget over time
NUM_SLEEP_CYCLES = 4  # "sleep cycles" -- consecutive dream -> re-infer passes

# Langevin inference (Eq. 17). The paper's Bidirectional Replay Protocol
# matches step budget T and noise sigma across wake/sleep, varying only
# prior strength beta.
# TODO, play around with the values below (LANGEVIN_STEPS, LANGEVIN_ETA, SLEEP CYCLES; don't change all of them too much at the same time though)
LANGEVIN_STEPS = 10     # T, per paper
LANGEVIN_ETA   = 0.01   # gradient step size -- NOT given a value in the
                         # paper; experimental knob, tune once you see output
LANGEVIN_SIGMA = 0.05   # injected noise std, per paper

WAKE_BETA  = 0.001   # weak prior -- high fidelity to the real frame
# TODO play around with this value (e.g. 0.3, 1, 2). Determines how much is "forgotten"
SLEEP_BETA = 0.1     # strong prior -- forces compression during dreaming

# Also save the wake state and every sleep cycle (as wake/, sleep_01/, ...
# subfolders) instead of only the final consolidated frame. Useful for a
# first look at what each pass is actually doing; costly for a full clip.
SAVE_INTERMEDIATE = True

SEED = 0

# ────────────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frame(path: Path, input_size) -> torch.Tensor:
    """input_size: int (square, old checkpoints) or (height, width) tuple."""
    height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
    img = cv2.imread(str(path))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)  # cv2 takes (w, h)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def save_frame(x: torch.Tensor, path: Path):
    img = (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), img)


def free_energy(x: torch.Tensor, z: torch.Tensor, decoder, beta: float) -> torch.Tensor:
    """F(x, z) = ||sigmoid(Dec(z)) - x||^2 + beta * ||z||^2, per-sample, summed for backward."""
    recon = torch.sigmoid(decoder(z))
    recon_term = (recon - x).pow(2).flatten(1).sum(dim=1)
    prior_term = beta * z.pow(2).sum(dim=1)
    return (recon_term + prior_term).sum()


def langevin_infer(x: torch.Tensor, z: torch.Tensor, decoder, beta: float, steps: int) -> torch.Tensor:
    """Settle z against sensory target x under prior strength beta (Eq. 17)."""
    z = z.detach()
    for _ in range(steps):
        z = z.detach().requires_grad_(True)
        loss = free_energy(x, z, decoder, beta)
        grad, = torch.autograd.grad(loss, z)
        with torch.no_grad():
            z = z - LANGEVIN_ETA * grad + LANGEVIN_SIGMA * torch.randn_like(z)
    return z.detach()


def consolidate_batch(x: torch.Tensor, model, save_paths: list, intermediate_dir: Path) -> torch.Tensor:
    """Runs wake + NUM_SLEEP_CYCLES sleep cycles on a batch of frames. Returns the final z."""
    with torch.no_grad():
        z = model.encode(x)

    # Wake: settle against the real frame, weak prior.
    z = langevin_infer(x, z, model.decoder, WAKE_BETA, LANGEVIN_STEPS)

    if SAVE_INTERMEDIATE:
        wake_dir = intermediate_dir / "wake"
        wake_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            wake_frames = torch.sigmoid(model.decode(z))
        for i, p in enumerate(save_paths):
            save_frame(wake_frames[i], wake_dir / p.name)

    # Sleep: repeatedly dream and re-settle under a strong prior.
    for cycle in range(1, NUM_SLEEP_CYCLES + 1):
        with torch.no_grad():
            dream = torch.sigmoid(model.decode(z))
        z = langevin_infer(dream, z, model.decoder, SLEEP_BETA, LANGEVIN_STEPS)

        if SAVE_INTERMEDIATE:
            cycle_dir = intermediate_dir / f"sleep_{cycle:02d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                cycle_frames = torch.sigmoid(model.decode(z))
            for i, p in enumerate(save_paths):
                save_frame(cycle_frames[i], cycle_dir / p.name)

    return z


def main():
    torch.manual_seed(SEED)

    base = Path(__file__).parent
    device = get_device()

    ckpt = torch.load(base / CHECKPOINT, map_location=device)
    model = PredictiveForgettingAutoencoder(
        in_channels=ckpt["in_channels"],
        channels=ckpt["channels"],
        latent_dim=ckpt["latent_dim"],
        input_size=ckpt["input_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    frame_paths = sorted((base / FRAMES_DIR).glob(FRAME_GLOB))
    if not frame_paths:
        print(f"No frames found matching '{FRAME_GLOB}' in {base / FRAMES_DIR}")
        return
    if TEST_FRAMES > 0:
        frame_paths = frame_paths[TEST_FRAME_START:TEST_FRAME_START + TEST_FRAMES]

    out_dir = base / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Predictive forgetting -- wake/sleep consolidation")
    print(f"  {len(frame_paths)} frames · device={device}")
    print(f"  {NUM_SLEEP_CYCLES} sleep cycles · {LANGEVIN_STEPS} Langevin steps/phase")
    print(f"  wake_beta={WAKE_BETA} · sleep_beta={SLEEP_BETA}\n")

    input_size = ckpt["input_size"]
    n = len(frame_paths)

    for start in range(0, n, BATCH_SIZE):
        batch_paths = frame_paths[start:start + BATCH_SIZE]
        x = torch.stack([load_frame(p, input_size) for p in batch_paths]).to(device)

        z_final = consolidate_batch(x, model, batch_paths, out_dir)

        with torch.no_grad():
            final_frames = torch.sigmoid(model.decode(z_final))
        for i, p in enumerate(batch_paths):
            save_frame(final_frames[i], out_dir / p.name)

        done = min(start + BATCH_SIZE, n)
        print(f"  {done}/{n}  ({done / n * 100:.0f}%)")

    print(f"\nDone. Consolidated frames in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
