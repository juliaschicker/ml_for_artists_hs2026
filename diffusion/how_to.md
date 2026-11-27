# 🦋🦋🦋 Diffusion model 🦋🦋🦋

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.
> - Grey boxes **without** `>>>` show how a line in a Python file should look.
>   Change that line in Sublime, then save the file.

---

# Part 1: Butterflies

## Preparation

**1.** For the first try, we'll work with a dataset of butterflies. The script downloads it automatically.

## Training

**2.** First, we'll check how long a training takes. In Sublime, open the file `training_config.py`.

**3.** Look for `num_epochs` and set it to 2:

```python
num_epochs = 2
```

**4.** Save the file.

**5.** Close all the programmes on your computer that you don't need right now. Keep only Sublime and the terminal open.

**6.** Your terminal has to be in the `diffusion` folder, and the venv has to be active: there should be `(venv)` at the beginning of the line in your terminal. Then run:

```bash
>>> python diffusion_model_training.py
```

This starts your training.

**7.** Wait until the training is over.

**8.** Check how long the training took.

> ⚠️ **If** each epoch took **less than 20 seconds**, let me know. We can increase the resolution.
>
> ⚠️ **If** each epoch took **more than 60 seconds**, let me know.

**9.** Go to `training_config.py` and set:

```python
num_epochs = 50
```

**10.** Save the file.

**11.** In your terminal, run:

```bash
>>> python diffusion_model_training.py
```

## Image generation

**1.** When the model has finished training, run:

```bash
>>> python generate_images.py
```

This creates a folder `generated` with the images.

**2.** To make more images, run:

```bash
>>> python generate_images.py --seed 1
```

```bash
>>> python generate_images.py --seed 2
```

```bash
>>> python generate_images.py --seed 3
```

The number after `--seed` has to change to make new images. A different seed number means the model starts from a different noise when making the images.

---

# Part 2: Your own dataset

> ⚠️ Don't try to increase the resolution (e.g. from 64 to 128) unless you have a really good computer.

## Preparation

**1.** Move the images in the `generated` folder into a subfolder named `butterflies`. Otherwise, we'll overwrite them later.

**2.** Prepare your dataset (min. 400 images, max. 1'000).

**3.** To make sure your dataset is not too big, go to your folder in the terminal with `cd`, like this:

```bash
>>> cd my-data-0
```

**4.** Make a new folder by typing this in your terminal:

```bash
>>> mkdir dataset_256
```

**5.** Make all the images smaller. This makes the training faster.

```bash
>>> sips -Z 256 dataset/*.jpg --out dataset_256/
```

## Training

**1.** Open `training_config.py` and change these two values:

```python
num_epochs = 5
```

We'll test again first whether we're really doing the right thing, so you don't waste as much time as I did.

```python
output_dir = "ddpm-my-data-0-64"
```

Change this to match the name of your folder with the new training data. The `64` says how big the images will be at the end.

> ⚠️ It's important to change this name. Otherwise, you will overwrite the butterfly model!

**2.** Open `diffusion_model_training.py` and go to **line 34**. Put a `#` in front of it, so it turns grey:

```python
# dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
```

This means the line is not read anymore, so the butterflies are no longer loaded.

Now delete the `#` in front of **line 36**, so it looks like this:

```python
dataset = load_dataset("imagefolder", data_dir="my-data-0/dataset_256", split="train")
```

Now the script loads your data from `my-data-0/dataset_256`. If your data is in a different folder, change the path accordingly.

**3.** Run a short test to check that everything is in place. In your terminal, run:

```bash
>>> python diffusion_model_training.py
```

**4.** When the five test epochs have run, generate images:

```bash
>>> python generate_images.py --model ddpm-my-data-0-64
```

> ⚠️ The name after `--model` must be the same as your `output_dir` in `training_config.py`.

**5.** Check the images: do they come from the right training data?

**6.** If everything is OK, go to `training_config.py` and change:

```python
num_epochs = 50
```

- If you have fewer than 1'000 images, set the epochs according to the table at the bottom of this file.
- If your computer is very slow, go a little below the recommended epochs, so you don't have to train for that long.

## Image generation

**1.** Make sure there are no images in the `generated` folder. Otherwise, they will be overwritten.

**2.** `python generate_images.py --seed 1` uses the butterfly model. For your own model, use this instead:

```bash
>>> python generate_images.py --model ddpm-my-data-0-64 --seed 1
```

> ⚠️ The name after `--model` must match the name of your model folder.

**3.** Check the results.

---

## How many epochs?

The fewer images you have, the more epochs you need. Together, they should add up to about 6'000 training steps.

| Images | Epochs | Steps per epoch (batch 8) | Total steps | Suggested `save_image_epochs` | What to expect |
|-------:|-------:|--------------------------:|------------:|------------------------------:|----------------|
| 100    | ~500   | 13                        | ~6'500      | 100                           | mostly reproduces the training images (memorises them) |
| 250    | ~200   | 32                        | ~6'400      | 50                            | close variations of the training images |
| 400    | ~125   | 50                        | ~6'250      | 25                            | starts to mix and recombine |
| 500    | ~100   | 63                        | ~6'300      | 25                            | |
| 750    | ~65    | 94                        | ~6'100      | 15                            | |
| 1'000  | 50     | 125                       | 6'250       | 10                            | new combinations (like the butterflies) |
