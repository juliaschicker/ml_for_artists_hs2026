<!-- Open this file on github: https://github.com/juliaschicker/ml_for_artists_hs2026/blob/main/gan/how_to.md -->
# Generative Adversarial Networks (GAN)

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.
> - Grey boxes **without** `>>>` show how a line in a Python file should look.
>   Change that line in Sublime, then save the file.

---

## Preparation

**1.** Decide whether you're using your own dataset or one from Hugging Face. A dataset should have between 1'000 and 10'000 images.

**2.** If you're using a dataset from Hugging Face, check the file `training_sets.md`. There you find a list of datasets and how to find more.

**3.** If the venv is not active in your terminal, activate it (see `venv.md`).

**4.** In your terminal, go to the `gan` folder (see `terminal_how_to.md` for more info).

## Training

**1.** **Using a dataset from Hugging Face?** Open `train_gan.py` and follow the instructions on **line 34**. Make sure this line has the name of the dataset you'd like to load:

```python
DATA_DIR = "huggingface:huggan/metfaces"
```

Change `huggan/metfaces` to the name of your dataset.

**2.** **Using your own data?** Open `train_gan.py` and make sure this line matches the folder with your dataset:

```python
DATA_DIR = "my-data"
```

- The folder has to be in the `gan` folder, next to `train_gan.py`.
- If your folder isn't called `my-data`, either rename your folder to `my-data`, or change the line so it's spelled **exactly** like your folder.

**3.** In `train_gan.py`, **line 38**, change `gan-my-data-64` to the name you'd like to give your GAN:

```python
OUTPUT_DIR = "gan-my-data-64"
```

**4.** We're ready to train! In your terminal, make sure the venv is active and you're in the `gan` folder. Then run:

```bash
>>> python train_gan.py
```

## Fine-tune / continue training

**1.** When your model has finished and you'd like to continue training it, or fine-tune it with other data, go to `train_gan.py`, **line 59**:

```python
RESUME_FROM = None
```

Set it to the model you'd like to continue from, e.g.:

```python
RESUME_FROM = "gan-metfaces-64/gan_model.pt"
```

- `gan-metfaces-64` is the folder the model is in. Check in the Finder what it's called in your case.
- `gan_model.pt` is the file of the GAN. Open the folder of your GAN in the Finder, and you'll see this file.

> ⚠️ Also change `OUTPUT_DIR` to a new name. Otherwise, your old model is overwritten.

**2.** To continue from a snapshot (an earlier stage of the training), change `RESUME_FROM` to this:

```python
RESUME_FROM = "gan-metfaces-64/snapshots/generator_0050.pt"
```

In the Finder, open the folder that was created during training (e.g. `gan-metfaces-64`). In there, you'll find a folder `snapshots` with `generator_0050.pt`, etc.

## Generating images

**1.** When your training has finished, you can generate images. In your terminal, run:

```bash
>>> python generate_gan.py --model gan-metfaces-64/gan_model.pt
```

- Make sure `gan-metfaces-64` matches the folder that was created during your training.
- A new folder `generated` is created (if it doesn't exist yet). That's where you find your results.

**2.** To get more images, add `--seed` and a number to the command above:

```bash
>>> python generate_gan.py --model gan-metfaces-64/gan_model.pt --seed 1
```

```bash
>>> python generate_gan.py --model gan-metfaces-64/gan_model.pt --seed 48
```

```bash
>>> python generate_gan.py --model gan-metfaces-64/gan_model.pt --seed 93
```

and so on.

**3.** To make a **walk** through the latent space (slowly moving from one image to the next), add `--walk` at the end of the command:

```bash
>>> python generate_gan.py --model gan-metfaces-64/gan_model.pt --walk
```

When it's finished, the very last line in your terminal looks something like this:

```bash
ffmpeg -framerate 25 -i 'generated/walk_seed0/frame-%05d.png' -c:v libx264 -pix_fmt yuv420p 'generated/walk_seed0/walk.mp4'
```

Copy it from your terminal, paste it, and press Enter. This makes a video called `walk.mp4`, in the `generated` folder, inside `walk_seed0`.

> 💡 With `--seed`, the walk visits other faces, and the folder is named after the seed: `--seed 1` → `walk_seed1`.
