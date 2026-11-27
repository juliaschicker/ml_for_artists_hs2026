# Train a GAN (DCGAN) on your own images, at 64 x 64 pixels.
# Based on the PyTorch DCGAN tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#
# How a GAN works -- two networks play against each other:
#   - the GENERATOR gets 100 random numbers (the "latent vector") and turns them into an image
#   - the DISCRIMINATOR gets an image and guesses: real (from your dataset) or fake (from the generator)?
# The discriminator learns to catch fakes, the generator learns to fool the discriminator.
# If both stay roughly equally good, the generator's images become more and more like your dataset.
#
# Run from this folder:   python train_gan.py
# Everything is saved in OUTPUT_DIR: sample grids, a loss plot, and the model (gan_model.pt).

# These imports make sure you have the necessary libraries
# pathlib: works with folders and file paths
from pathlib import Path
# random: Python's own random numbers (only used for the seed)
import random

# PyTorch: tensors, neural networks, optimizers, GPU support
import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision: image tools (resizing, cropping) and saving image grids
from torchvision import transforms
from torchvision.utils import save_image
# Pillow: opens the image files
from PIL import Image
# tqdm: progress bar in the terminal
from tqdm.auto import tqdm


# ----- Settings -----
# DATA_DIR = "my-data"            # folder with your training images (subfolders are included)
# To train on a Hugging Face dataset instead: put a # in front of the line above and remove the # below.
# "huggingface:" + the dataset name as shown on huggingface.co (downloaded automatically the first time).
DATA_DIR = "huggingface:huggan/metfaces"
MAX_IMAGES = None               # None = use all images. A number (e.g. 5000) = use a random selection (for big datasets)
OUTPUT_DIR = "gan-metfaces-64-300"   # folder the model, samples and loss plot are saved to

NUM_EPOCHS = 200                # how often the whole dataset is shown. Aim for ~5'000-10'000 steps in total:
                                # 1'000 images -> ~300-500 epochs, 5'000 images -> ~100 epochs
BATCH_SIZE = 64                 # images per training step
SAVE_IMAGE_EPOCHS = 10          # save a sample grid every 10 epochs
SAVE_MODEL_EPOCHS = 50          # keep a copy of the generator every 50 epochs (in OUTPUT_DIR/snapshots)

AUGMENT = True                  # DiffAugment: random colour, shift and cut-out during training.
                                # Strongly recommended below ~5'000 images, otherwise the GAN collapses quickly.
FLIP = True                     # also use mirrored versions of the images

LEARNING_RATE = 2e-4            # the tutorial's value; lower (e.g. 1e-4) if training becomes unstable
BETA1 = 0.5                     # optimizer setting from the DCGAN paper (leave it)

LATENT_SIZE = 100               # how many random numbers the generator starts from
GENERATOR_FEATURES = 64         # size of the generator network (64 = tutorial; 32 = smaller/faster)
DISCRIMINATOR_FEATURES = 64     # size of the discriminator network

# Continue training an existing model: set to its file, e.g. "gan-my-data-64/gan_model.pt".
# Works with new images too (fine-tuning). Use a NEW OUTPUT_DIR, so the old model is not overwritten.
RESUME_FROM = "gan-metfaces-64/gan_model.pt"

SEED = 0

IMAGE_SIZE = 64                 # fixed: the network below is built for 64 x 64
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----- The two networks (unchanged from the tutorial, except: no Sigmoid at the end of the discriminator) -----
class Generator(nn.Module):
    """100 random numbers -> 64 x 64 RGB image. Each step doubles the image size: 4 -> 8 -> 16 -> 32 -> 64."""

    def __init__(self, latent_size=LATENT_SIZE, features=GENERATOR_FEATURES):
        super().__init__()
        f = features
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, f * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(f * 8), nn.ReLU(True),  # 4 x 4
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 4), nn.ReLU(True),        # 8 x 8
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 2), nn.ReLU(True),        # 16 x 16
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1, bias=False), nn.BatchNorm2d(f), nn.ReLU(True),                # 32 x 32
            nn.ConvTranspose2d(f, 3, 4, 2, 1, bias=False), nn.Tanh(),                                           # 64 x 64, values -1...1
        )

    def forward(self, z):
        return self.net(z.view(z.shape[0], -1, 1, 1))


class Discriminator(nn.Module):
    """64 x 64 RGB image -> one number: high = "looks real", low = "looks fake"."""

    def __init__(self, features=DISCRIMINATOR_FEATURES):
        super().__init__()
        f = features
        self.net = nn.Sequential(
            nn.Conv2d(3, f, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),                              # 32 x 32
            nn.Conv2d(f, f * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 2), nn.LeakyReLU(0.2, inplace=True),   # 16 x 16
            nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 4), nn.LeakyReLU(0.2, inplace=True),  # 8 x 8
            nn.Conv2d(f * 4, f * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 8), nn.LeakyReLU(0.2, inplace=True),  # 4 x 4
            nn.Conv2d(f * 8, 1, 4, 1, 0, bias=False),                                                           # 1 number
        )

    def forward(self, image):
        return self.net(image).view(-1)


def init_weights(module):
    """The DCGAN paper starts all weights as small random numbers (mean 0, spread 0.02)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


# ----- DiffAugment (Zhao et al. 2020): the same random changes for real AND fake images -----
# The discriminator then can't simply memorise your few training images, and because the changes are
# applied to the fakes too, the generator does NOT learn to produce shifted or recoloured images.
def augment(x):
    n, device = x.shape[0], x.device
    # colour: random brightness, saturation and contrast per image
    x = x + (torch.rand(n, 1, 1, 1, device=device) - 0.5)
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * (torch.rand(n, 1, 1, 1, device=device) * 2) + mean
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * (torch.rand(n, 1, 1, 1, device=device) + 0.5) + mean
    # translation: shift each image by up to 1/8 of its size (the empty part is filled with zeros)
    shift = (torch.rand(n, 2, device=device) - 0.5) * 0.25 * 2   # in grid units: the image spans -1...1
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    # cut-out: black out a random square of half the image size
    size = x.shape[2] // 2
    ys = torch.arange(x.shape[2], device=device).view(1, -1, 1)
    xs = torch.arange(x.shape[3], device=device).view(1, 1, -1)
    top = torch.randint(-size // 2, x.shape[2] - size // 2, (n, 1, 1), device=device)
    left = torch.randint(-size // 2, x.shape[3] - size // 2, (n, 1, 1), device=device)
    keep = ~((ys >= top) & (ys < top + size) & (xs >= left) & (xs < left + size))
    return x * keep.unsqueeze(1)


# ----- Loading the images -----
def to_rgb(image):
    """Converts any image to RGB. Transparent parts (e.g. PNGs) become white instead of random colours."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def load_images(source, base):
    """Loads every image once, crops it to a square and resizes it to 64 x 64. Kept in memory for speed.
    source is either a folder (relative to this script) or "huggingface:<dataset name>"."""
    prepare = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),        # shortest side becomes 64, keeps proportions
        transforms.CenterCrop(IMAGE_SIZE),    # cut a 64 x 64 square from the middle
        transforms.PILToTensor(),             # numbers 0...255
    ])
    images = []

    if source.startswith("huggingface:"):
        # Hugging Face datasets: downloads the dataset (only the first time) and loads it
        from datasets import load_dataset, Image as ImageColumn
        name = source[len("huggingface:"):]
        print(f"Loading the Hugging Face dataset '{name}' (the first time it is downloaded, this can take a while) ...")
        dataset = load_dataset(name, split="train")
        image_columns = [column for column, kind in dataset.features.items() if isinstance(kind, ImageColumn)]
        if not image_columns:
            print(f"'{name}' has no image column. Columns: {list(dataset.features)}")
            return None
        if MAX_IMAGES and MAX_IMAGES < len(dataset):
            dataset = dataset.shuffle(seed=SEED).select(range(MAX_IMAGES))
        for example in tqdm(dataset, desc="Loading images"):   # one by one, so big images don't fill the memory
            images.append(prepare(to_rgb(example[image_columns[0]])))
    else:
        paths = sorted(p for p in (base / source).rglob("*") if p.suffix.lower() in IMAGE_TYPES)
        if MAX_IMAGES and MAX_IMAGES < len(paths):
            paths = sorted(random.Random(SEED).sample(paths, MAX_IMAGES))
        for path in tqdm(paths, desc="Loading images"):
            with Image.open(path) as image:
                images.append(prepare(to_rgb(image)))

    return torch.stack(images) if images else None


def to_model_range(batch):
    """0...255 -> -1...1 (what the networks work with)."""
    return batch.float() / 127.5 - 1


def save_loss_plot(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(history["discriminator"], label="discriminator loss")
    plt.plot(history["generator"], label="generator loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple GPU
    if torch.cuda.is_available():
        return torch.device("cuda")     # NVIDIA GPU
    return torch.device("cpu")


# ----- Training -----
def main():
    base = Path(__file__).parent
    output_dir = base / OUTPUT_DIR
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()

    images = load_images(DATA_DIR, base)
    if images is None:
        print(f"No images found in {DATA_DIR}")
        return
    if len(images) < BATCH_SIZE:
        print(f"Only {len(images)} images found, but BATCH_SIZE is {BATCH_SIZE}. Use more images or a smaller BATCH_SIZE.")
        return
    steps_per_epoch = len(images) // BATCH_SIZE
    print(f"{len(images)} images, {steps_per_epoch} steps per epoch, device: {device}")
    if len(images) < 1000:
        print("  note: below ~1'000 images GANs often collapse (all outputs look the same). Keep AUGMENT = True.")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    history = {"discriminator": [], "generator": []}
    start_epoch = 0

    if RESUME_FROM:
        checkpoint = torch.load(base / RESUME_FROM, map_location=device)
        if (checkpoint["latent_size"], checkpoint["generator_features"], checkpoint["discriminator_features"]) != \
                (LATENT_SIZE, GENERATOR_FEATURES, DISCRIMINATOR_FEATURES):
            raise ValueError("LATENT_SIZE / GENERATOR_FEATURES / DISCRIMINATOR_FEATURES must be the same as in the model you resume from.")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        history = checkpoint.get("history", history)
        start_epoch = checkpoint["epoch"]
        print(f"Continuing from '{RESUME_FROM}' (trained for {start_epoch} epochs so far)")
    else:
        generator.apply(init_weights)
        discriminator.apply(init_weights)
        print("Training a new GAN from scratch")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "snapshots").mkdir(exist_ok=True)

    # The same 64 latent vectors for every sample grid, so you can watch them develop over time
    fixed_noise = torch.randn(64, LATENT_SIZE, generator=torch.Generator().manual_seed(SEED)).to(device)
    loss_fn = nn.BCEWithLogitsLoss()   # "real or fake?" as a yes/no question

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        order = torch.randperm(len(images))
        sum_d, sum_g = 0.0, 0.0
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{start_epoch + NUM_EPOCHS}", leave=False)

        for step in progress:
            real = to_model_range(images[order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]).to(device)
            if FLIP:
                flip = torch.rand(real.shape[0], device=device) < 0.5
                real = torch.where(flip.view(-1, 1, 1, 1), real.flip(3), real)
            noise = torch.randn(real.shape[0], LATENT_SIZE, device=device)
            fake = generator(noise)
            show = augment if AUGMENT else (lambda x: x)

            # 1) Discriminator: learn to say "real" (1) to real images and "fake" (0) to generated ones
            optimizer_d.zero_grad()
            out_real = discriminator(show(real))
            out_fake = discriminator(show(fake.detach()))
            loss_d = loss_fn(out_real, torch.ones_like(out_real)) + loss_fn(out_fake, torch.zeros_like(out_fake))
            loss_d.backward()
            optimizer_d.step()

            # 2) Generator: learn to make the discriminator say "real" to generated images
            optimizer_g.zero_grad()
            out_fake = discriminator(show(fake))
            loss_g = loss_fn(out_fake, torch.ones_like(out_fake))
            loss_g.backward()
            optimizer_g.step()

            sum_d += loss_d.item()
            sum_g += loss_g.item()
            progress.set_postfix(d_loss=f"{loss_d.item():.2f}", g_loss=f"{loss_g.item():.2f}")

        history["discriminator"].append(sum_d / steps_per_epoch)
        history["generator"].append(sum_g / steps_per_epoch)
        done = epoch + 1
        print(f"Epoch {done}: discriminator loss {history['discriminator'][-1]:.2f}, generator loss {history['generator'][-1]:.2f}")

        last_epoch = done == start_epoch + NUM_EPOCHS
        if done % SAVE_IMAGE_EPOCHS == 0 or last_epoch:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            generator.train()
            save_image(samples, output_dir / "samples" / f"{done:04d}.png", nrow=8, normalize=True, value_range=(-1, 1))
            save_loss_plot(history, output_dir / "losses.png")

        if done % SAVE_MODEL_EPOCHS == 0 or last_epoch:
            settings = {"latent_size": LATENT_SIZE, "generator_features": GENERATOR_FEATURES,
                        "discriminator_features": DISCRIMINATOR_FEATURES, "epoch": done}
            # everything needed to continue training (overwritten each time)
            torch.save({**settings, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                        "history": history}, output_dir / "gan_model.pt")
            # a small copy of only the generator, kept for every SAVE_MODEL_EPOCHS (GANs can get worse again!)
            torch.save({**settings, "generator": generator.state_dict()}, output_dir / "snapshots" / f"generator_{done:04d}.pt")

    print(f"Done. Model, samples and loss plot are in {output_dir}")


if __name__ == "__main__":
    main()
