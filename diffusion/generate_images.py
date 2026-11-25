import argparse
import os
import torch
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid

# Settings you can change from the terminal, e.g. --num 8 --seed 3
parser = argparse.ArgumentParser(description="Generate butterflies with the trained model")
parser.add_argument("--num", type=int, default=8, help="how many images to generate")
parser.add_argument("--seed", type=int, default=0, help="same seed = same images, new seed = new images")
parser.add_argument("--steps", type=int, default=1000, help="denoising steps: 1000 = best quality, 250 = about 4x faster and still good")
parser.add_argument("--model", default="ddpm-butterflies-64", help="folder of the trained model")
parser.add_argument("--out", default="generated", help="folder to save the images in")
args = parser.parse_args()

# Load the trained model (UNet + noise scheduler) from the training output folder
pipeline = DDPMPipeline.from_pretrained(args.model)

# Use the Apple GPU if available, otherwise the CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipeline.to(device)

# Generate: start from pure noise and remove it step by step
generator = torch.Generator(device="cpu").manual_seed(args.seed)
images = pipeline(batch_size=args.num, generator=generator, num_inference_steps=args.steps).images

# Save each image separately, file names include the seed so nothing gets overwritten
os.makedirs(args.out, exist_ok=True)
for i, image in enumerate(images):
    image.save(f"{args.out}/seed{args.seed}_{i:02d}.png")
make_image_grid(images, rows=1, cols=args.num).save(f"{args.out}/seed{args.seed}_grid.png")
print(f"Saved {args.num} butterflies to {args.out}/")
