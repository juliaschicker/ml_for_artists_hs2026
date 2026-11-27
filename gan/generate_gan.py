# Generate images with a trained GAN -- single images or a "latent walk" video.
#
# Every GAN image comes from 100 random numbers (a point in the latent space).
#   images mode: picks random points and saves the images
#   walk mode:   glides smoothly from one random point to the next and saves every in-between step as a frame
#
# Run from this folder, e.g.:
#   python generate_gan.py                               (16 images with seed 0)
#   python generate_gan.py --num 64 --seed 3             (64 other images)
#   python generate_gan.py --walk                        (frames for a latent walk video)
#   python generate_gan.py --model gan-my-data-64/snapshots/generator_0150.pt   (an earlier stage of the training)

# argparse: reads the settings you type after "python generate_gan.py"
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
# the generator network from the training script (importing it does NOT start a training)
from train_gan import Generator, get_device


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gan-my-data-64/gan_model.pt", help="the trained model file")
parser.add_argument("--out", default="generated", help="folder for the results")
parser.add_argument("--num", type=int, default=16, help="how many images")
parser.add_argument("--seed", type=int, default=0, help="change it to get different images")
parser.add_argument("--size", type=int, default=256, help="saved image size in pixels (the GAN itself makes 64 x 64)")
parser.add_argument("--walk", action="store_true", help="make frames for a latent walk instead of single images")
parser.add_argument("--points", type=int, default=8, help="walk: how many random points to visit")
parser.add_argument("--frames", type=int, default=50, help="walk: frames between two points (50 = 2 seconds at 25 fps)")
args = parser.parse_args()

base = Path(__file__).parent
device = get_device()

# Load the generator
checkpoint = torch.load(base / args.model, map_location=device)
generator = Generator(checkpoint["latent_size"], checkpoint["generator_features"]).to(device)
generator.load_state_dict(checkpoint["generator"])
generator.eval()
print(f"Loaded '{args.model}' (trained for {checkpoint['epoch']} epochs)")

random_numbers = torch.Generator().manual_seed(args.seed)


def make_images(latents):
    """Latent vectors -> images in 0...1, resized to --size."""
    with torch.no_grad():
        images = generator(latents.to(device))
    images = (images.clamp(-1, 1) + 1) / 2
    if args.size != 64:
        images = torch.nn.functional.interpolate(images, size=(args.size, args.size), mode="bicubic", align_corners=False).clamp(0, 1)
    return images.cpu()


def slerp(a, b, t):
    """Moves along a curve between two latent points (better than a straight line for random noise)."""
    a_unit, b_unit = a / a.norm(), b / b.norm()
    angle = torch.acos((a_unit * b_unit).sum().clamp(-1, 1))
    if angle.abs() < 1e-4:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * angle) * a + torch.sin(t * angle) * b) / torch.sin(angle)


if not args.walk:
    out_dir = base / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    latents = torch.randn(args.num, checkpoint["latent_size"], generator=random_numbers)
    images = make_images(latents)
    for i, image in enumerate(images):
        save_image(image, out_dir / f"seed{args.seed}_{i:03d}.png")
    save_image(images, out_dir / f"seed{args.seed}_grid.png", nrow=8)
    print(f"Saved {args.num} images and a grid in {out_dir}")

else:
    out_dir = base / args.out / f"walk_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    points = torch.randn(args.points, checkpoint["latent_size"], generator=random_numbers)
    frame = 0
    for i in range(args.points):
        start, end = points[i], points[(i + 1) % args.points]   # the last point leads back to the first: a seamless loop
        steps = torch.stack([slerp(start, end, t) for t in torch.linspace(0, 1, args.frames + 1)[:-1]])
        for image in make_images(steps):
            save_image(image, out_dir / f"frame-{frame:05d}.png")
            frame += 1
    print(f"Saved {frame} frames in {out_dir}")
    print("To make a video, run:")
    print(f"  ffmpeg -framerate 25 -i '{out_dir.relative_to(base)}/frame-%05d.png' -c:v libx264 -pix_fmt yuv420p "
          f"'{out_dir.relative_to(base)}/walk.mp4'")
