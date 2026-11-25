# Pick random images from all subfolders, shrink them, and copy them into one training folder.
#
# Place this script in a folder whose subfolders contain images, then run:
#   python pick_images.py
#
# The original images are only READ, never changed, moved or deleted.
# The shrunk copies are saved in OUTPUT_DIR (next to this script).

import random
from pathlib import Path

# Pillow: opens, crops, resizes and saves images
from PIL import Image, ImageOps


# ----- Settings -----
COUNT = 750                    # how many images to pick
SIZE = (256, 144)              # (width, height) of the copies in pixels
FIT = "crop"                   # "crop" = cut images with another shape (e.g. 4:3 or portrait) to 16:9 from the centre
                               # "stretch" = squeeze them into 256 x 144 (distorts them)
OUTPUT_DIR = "training-data"
SEED = 0                       # same seed = same random pick; change it for a different selection
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    base = Path(__file__).parent
    output_dir = base / OUTPUT_DIR

    # Don't mix a new pick with an old one
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"'{OUTPUT_DIR}' already exists and is not empty.")
        print("Rename or delete it first (or change OUTPUT_DIR), then run the script again.")
        return

    # Find all images in all subfolders (but not in the output folder itself)
    images = [
        p for p in sorted(base.rglob("*"))
        if p.suffix.lower() in IMAGE_TYPES
        and p.parent != base                       # only images inside subfolders
        and output_dir not in p.parents            # skip the output folder
    ]
    print(f"Found {len(images)} images in the subfolders.")
    if not images:
        return

    # Pick at random
    random.seed(SEED)
    if len(images) < COUNT:
        print(f"That's fewer than {COUNT}, so all {len(images)} are used.")
        picked = images
    else:
        picked = random.sample(images, COUNT)

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(picked, start=1):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")   # apply phone rotation, drop transparency
            if FIT == "crop":
                image = ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS)   # crop to shape, then shrink
            else:
                image = image.resize(SIZE, Image.Resampling.LANCZOS)

            # Name = subfolder(s) + original name, so equal file names from different subfolders don't collide
            relative = path.relative_to(base).with_suffix("")
            new_name = "_".join(relative.parts) + ".jpg"
            image.save(output_dir / new_name, quality=95)

        if i % 100 == 0:
            print(f"  {i}/{len(picked)} copied")

    print(f"Done: {len(picked)} images saved to '{OUTPUT_DIR}' at {SIZE[0]} x {SIZE[1]} px.")


if __name__ == "__main__":
    main()
