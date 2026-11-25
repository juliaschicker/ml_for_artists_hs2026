# Merge two trained diffusion models into a new one by blending their weights.
#
# Every model is a long list of numbers (its weights). Merging takes both lists and mixes them:
#   merged = (1 - mix) * model A + mix * model B
# mix = 0 is pure model A, mix = 1 is pure model B, 0.5 is halfway in between.
#
# IMPORTANT: this only gives good results when both models come from the SAME starting model,
# e.g. you trained one base model and then fine-tuned it twice on different data
# (training_config.py -> resume_from). Two models trained separately from scratch have
# organised their weights completely differently; mixing those gives noise.
#
# Run from this folder:   python merge_models.py
# Each mix is saved as its own model folder (usable with generate_images.py --model ...),
# plus a sample grid so you can compare the mixes.

import os
import torch
# diffusers: the model class, the full pipeline (model + scheduler) and the image grid helper
from diffusers import UNet2DModel, DDPMPipeline
from diffusers.utils import make_image_grid


# ----- Settings -----
MODEL_A = "ddpm-my-data-0-64"          # folder of the first trained model
MODEL_B = "ddpm-my-data-1-64"          # folder of the second trained model
MIXES = [0.25, 0.5, 0.75]              # how much of model B goes into each merge (0 = only A, 1 = only B)
OUTPUT_NAME = "merged-data-0-data-1"   # merged models are saved as e.g. "merged-data-0-data-1-mix050"

MAKE_SAMPLES = True                    # also generate a sample grid for every mix
SAMPLE_SEED = 0                        # same seed for all mixes, so they are comparable
SAMPLE_COUNT = 8
SAMPLE_STEPS = 250                     # denoising steps for the samples (1000 = best, 250 = 4x faster)


def ignore_paths(model_config):
    """The model's settings without entries that don't describe its structure."""
    return {k: v for k, v in model_config.items() if not k.startswith("_")}


# Load both models
unet_a = UNet2DModel.from_pretrained(MODEL_A, subfolder="unet")
unet_b = UNet2DModel.from_pretrained(MODEL_B, subfolder="unet")

# Both models must have exactly the same structure, otherwise there's nothing to blend
if ignore_paths(unet_a.config) != ignore_paths(unet_b.config):
    raise ValueError(
        "The two models have a different structure (e.g. different image_size or block_out_channels) "
        "and cannot be merged."
    )

weights_a = unet_a.state_dict()
weights_b = unet_b.state_dict()
scheduler = DDPMPipeline.from_pretrained(MODEL_A).scheduler
device = "mps" if torch.backends.mps.is_available() else "cpu"

# The two original models, for comparison in the sample grids
sample_rows = []

for mix in MIXES:
    # Blend every weight: (1 - mix) * A + mix * B
    merged_weights = {name: (1 - mix) * weights_a[name] + mix * weights_b[name] for name in weights_a}

    merged_unet = UNet2DModel.from_config(unet_a.config)
    merged_unet.load_state_dict(merged_weights)

    out_dir = f"{OUTPUT_NAME}-mix{round(mix * 100):03d}"
    pipeline = DDPMPipeline(unet=merged_unet, scheduler=scheduler)
    pipeline.save_pretrained(out_dir)
    print(f"Saved mix {mix} -> {out_dir}")

    if MAKE_SAMPLES:
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        generator = torch.Generator("cpu").manual_seed(SAMPLE_SEED)
        images = pipeline(batch_size=SAMPLE_COUNT, generator=generator, num_inference_steps=SAMPLE_STEPS).images
        grid = make_image_grid(images, rows=1, cols=SAMPLE_COUNT)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        grid.save(os.path.join(out_dir, "samples", "merge_samples.png"))
        sample_rows.append(grid)

# One overview image with all mixes on top of each other (top row = smallest mix)
if MAKE_SAMPLES and sample_rows:
    overview = make_image_grid(sample_rows, rows=len(sample_rows), cols=1)
    overview.save(f"{OUTPUT_NAME}-overview.png")
    print(f"Overview of all mixes: {OUTPUT_NAME}-overview.png")
