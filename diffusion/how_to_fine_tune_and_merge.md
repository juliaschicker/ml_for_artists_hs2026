<!-- Open this file on github: https://github.com/juliaschicker/ml_for_artists_hs2026/blob/main/diffusion/how_to_fine_tune_and_merge.md -->

# Fine-tuning and merging diffusion models

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.
> - Grey boxes **without** `>>>` show how a line in a Python file should look.
>   Change that line in Sublime, then save the file.

---

# Part 1: Fine-tuning

## Preparation

**1.** You need a model that is already trained, e.g. the butterfly model or your own model.

**2.** You need a dataset that is different from the one your base model was trained on.

## Fine-tuning

**1.** In `training_config.py`, **line 15**, change

```python
output_dir = "ddpm-butterflies-64"
```

to something like

```python
output_dir = "ddpm-butterflies-fine-tune-64"
```

**2.** In `training_config.py`, **line 21**, change

```python
resume_from = None
```

to

```python
resume_from = "ddpm-butterflies-64"
```

or to the name of your own base model, if you'd like to start from that one.

**3.** In `training_config.py`, **line 8**, change

```python
num_epochs = 2
```

to

```python
num_epochs = 10
```

or whatever you'd like to experiment with. I'd suggest only a few epochs to start with.

**4.** In `diffusion_model_training.py`, change **line 36** so it points to the dataset you'd like to train with:

```python
dataset = load_dataset("imagefolder", data_dir="my-data-0/dataset_256", split="train")
```

**5.** In your terminal, start the fine-tuning:

```bash
>>> python diffusion_model_training.py
```

## Image generation

**1.** Make sure there are no images in the `generated` folder. Otherwise, they will be overwritten.

**2.** Generate images with your fine-tuned model:

```bash
>>> python generate_images.py --model ddpm-butterflies-fine-tune-64 --seed 1
```

> ⚠️ The name after `--model` must match the name of your model folder.

**3.** Check the results.

## Continue fine-tuning

**1.** To continue fine-tuning the model you just made, change `resume_from` to that model:

```python
resume_from = "ddpm-butterflies-fine-tune-64"
```

> ⚠️ Don't forget to also change `output_dir`. Otherwise, the model is overwritten.

The new `num_epochs` are added on top of the epochs you have already fine-tuned.

---

# Part 2: Merging models

Two models can be merged if they are **fine-tuned from the same base model**.

Two different base models can't be merged, because their weights are organised completely differently. The result is just colourful noise.

Merging takes the weights of both models (a long list of numbers) and mixes them: a certain amount of the first model's weights plus a certain amount of the second model's weights.

## Preparation

**1.** You need two models that were fine-tuned from the same base model. If you don't have that, you can split your own dataset in two and fine-tune the same base model on each half.

**2.** Move the images in the `generated` folder somewhere else.

## Merging

**1.** In `merge_models.py`, change **lines 24, 25 and 27**:

```python
MODEL_A = "ddpm-my-data-0-64"
MODEL_B = "ddpm-my-data-1-64"
OUTPUT_NAME = "merged-data-0-data-1"
```

- `MODEL_A` and `MODEL_B` are the two models you'd like to merge.
- `OUTPUT_NAME` is the name the merged models are saved under.

**2.** Run the script in your terminal:

```bash
>>> python merge_models.py
```

## Image generation

**1.** Make sure there are no images in the `generated` folder. Otherwise, they will be overwritten.

**2.** The script makes three different mixes (0.25, 0.5 and 0.75), each in its own folder: `OUTPUT_NAME` + `-mix025`, `-mix050`, `-mix075`. Generate images with one of them:

```bash
>>> python generate_images.py --model merged-data-0-data-1-mix025 --seed 1
```

> ⚠️ The name after `--model` must match the name of your merged model folder.

**3.** Check the results.
