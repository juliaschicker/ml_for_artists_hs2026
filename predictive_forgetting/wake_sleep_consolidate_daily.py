#!/usr/bin/env python3
"""
Daily consolidation loop -- an installation-oriented extension of
wake_sleep_consolidate.py. Instead of always starting over from the raw
footage, this script lets a fixed set of frames get progressively MORE
consolidated every time it's run (e.g. once a day): each run loads the
*previous* run's final latent z and keeps compressing from there.

The mechanism is the paper's own iterative-refinement idea (Section 2.4,
"Dynamic State Refinement vs. Static Compression"):

    "at retrieval: when a stored trace is accessed in a new context, it
    can undergo further compression informed by the current
    computational state"

and the Data-Processing-Inequality argument (Section 2.3) that each
consolidation step can only lose information relative to the last
(Fig. 1's Z0 -> Z1 -> ... -> Z5, here spread across runs instead of
within a single one). This day-by-day framing for a live installation
is our own extension -- the paper itself only demonstrates iterative
refinement within a single sitting, not across separate invocations.

As in wake_sleep_consolidate.py, Dec_Theta's (and Enc_Phi's) weights
are frozen throughout and CHECKPOINT is only ever read, never written
(Section 3.2: "The generative model weights are frozen throughout this
experiment"). The only thing that persists and grows between runs is a
small per-frame z state file (latent_dim floats/frame -- a few hundred
KB for a whole clip), not the model.

ENABLE_DAILY_WAKE toggles whether each day briefly re-anchors z against
the real source frame before sleeping further ("wake"), or whether it
dreams purely from its own last state with no further grounding after
day 1. Flipping it points at a separate state/output folder (suffixed
_wake / _nowake), so both modes can be run independently and compared.

Usage:
    python wake_sleep_consolidate_daily.py
    (run again whenever you want -- once a day, several times while
    testing, whatever -- each run advances one more "day".)
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from predictive_forgetting_model import PredictiveForgettingAutoencoder

# ─── PARAMETERS ─────────────────────────────────────────────────────────────

CHECKPOINT   = "predictive_forgetting_model_succ_14_260917_256_144.pt"

FRAMES_DIR   = "test_choreo_260917"
FRAME_GLOB   = "*.jpg"

STATE_DIR_BASE  = "daily_consolidation_state"
OUTPUT_DIR_BASE = "consolidated_predictive_forgetting_choreo_260917_succ_14_daily"

BATCH_SIZE   = 16     # frames are consolidated independently, batched for throughput

TEST_FRAMES  = 0      # >0 to process only a slice of N frames (quick test)
TEST_FRAME_START = 0

NUM_SLEEP_CYCLES_PER_DAY = 5   # sleep cycles added *per run* -- tune for pace

# Langevin inference (Eq. 17), same knobs as wake_sleep_consolidate.py.
LANGEVIN_STEPS = 10
LANGEVIN_ETA   = 0.01
LANGEVIN_SIGMA = 0.05

WAKE_BETA  = 0.001   # weak prior -- light re-anchor against the real frame
SLEEP_BETA = 0.1     # strong prior -- forces compression during dreaming

# True: each day does one light wake re-anchor (against the real source
# frame) before sleeping further. False: after day 1, it only ever dreams
# from its own last state -- no more grounding in the real footage at all.
# Flipping this changes STATE_DIR/OUTPUT_DIR below (suffixed _wake/_nowake),
# so both modes get independent, directly comparable trajectories.
ENABLE_DAILY_WAKE = True

# Off by default here (unlike wake_sleep_consolidate.py): this script runs
# repeatedly over many days, so per-run wake/sleep_NN intermediate subfolders
# would pile up fast. Flip on for one run if you want to inspect a single
# day's steps closely.
SAVE_INTERMEDIATE = False

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


def consolidate_batch(x: torch.Tensor, z: torch.Tensor, model, enable_wake: bool,
                       save_paths: list, intermediate_dir: Path) -> torch.Tensor:
    """
    One day's consolidation step on a batch: an optional wake re-anchor
    against the real frame, then NUM_SLEEP_CYCLES_PER_DAY sleep cycles,
    starting from the given z -- either freshly encoded (day 1) or loaded
    from the previous run's saved state (later days).
    """
    if enable_wake:
        z = langevin_infer(x, z, model.decoder, WAKE_BETA, LANGEVIN_STEPS)
        if SAVE_INTERMEDIATE:
            wake_dir = intermediate_dir / "wake"
            wake_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                wake_frames = torch.sigmoid(model.decode(z))
            for i, p in enumerate(save_paths):
                save_frame(wake_frames[i], wake_dir / p.name)

    for cycle in range(1, NUM_SLEEP_CYCLES_PER_DAY + 1):
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
    frame_names = [p.name for p in frame_paths]

    mode_tag = "wake" if ENABLE_DAILY_WAKE else "nowake"
    state_path = base / f"{STATE_DIR_BASE}_{mode_tag}" / "z_state.pt"
    out_base = base / f"{OUTPUT_DIR_BASE}_{mode_tag}"
    latest_dir = out_base / "latest"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_base / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    input_size = ckpt["input_size"]
    n = len(frame_paths)

    fresh_start = True
    z_prev = None
    day = 0
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        if state["frame_names"] == frame_names:
            z_prev = state["z"].to(device)
            day = state["day"]
            fresh_start = False
        else:
            print(f"  frame set in {FRAMES_DIR} differs from the saved state -- starting over as day 1")

    print(f"Predictive forgetting -- daily consolidation loop ({mode_tag})")
    print(f"  {n} frames · device={device}")
    print(f"  {'day 1 (fresh start)' if fresh_start else f'day {day + 1}'} · "
          f"{NUM_SLEEP_CYCLES_PER_DAY} sleep cycles this run · "
          f"daily wake re-anchor={'on' if ENABLE_DAILY_WAKE else 'off'}\n")

    new_z_chunks = []

    for start in range(0, n, BATCH_SIZE):
        batch_paths = frame_paths[start:start + BATCH_SIZE]
        x = torch.stack([load_frame(p, input_size) for p in batch_paths]).to(device)

        if fresh_start:
            with torch.no_grad():
                z = model.encode(x)
            enable_wake = True  # day 1 always needs one real grounding step
        else:
            z = z_prev[start:start + BATCH_SIZE]
            enable_wake = ENABLE_DAILY_WAKE

        z = consolidate_batch(x, z, model, enable_wake, batch_paths, run_dir)
        new_z_chunks.append(z.detach().cpu())

        with torch.no_grad():
            final_frames = torch.sigmoid(model.decode(z))
        for i, p in enumerate(batch_paths):
            save_frame(final_frames[i], run_dir / p.name)
            save_frame(final_frames[i], latest_dir / p.name)

        done = min(start + BATCH_SIZE, n)
        print(f"  {done}/{n}  ({done / n * 100:.0f}%)")

    new_day = 1 if fresh_start else day + 1
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    torch.save({
        "z": torch.cat(new_z_chunks, dim=0),
        "frame_names": frame_names,
        "day": new_day,
        "last_run": run_stamp,
        "mode": mode_tag,
    }, tmp_path)
    tmp_path.replace(state_path)

    cumulative_cycles = new_day * NUM_SLEEP_CYCLES_PER_DAY
    print(f"\nDay {new_day} done. Output: {run_dir.relative_to(base)}/ (and {latest_dir.relative_to(base)}/)")
    print(f"State saved to {state_path.relative_to(base)} -- "
          f"{cumulative_cycles} sleep cycles accumulated so far.")


if __name__ == "__main__":
    main()
