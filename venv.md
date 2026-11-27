# Setting up the venv

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.

---

## Before you start

In your terminal, make sure you're in the `ml_for_artists` folder. The text right before your cursor should look something like this:

```
yourname@your-computer ml_for_artists %
```

The last part is the folder you are in.

## Create and activate the venv

Run these commands one after another in your terminal.

**1.** Create the venv:

```bash
>>> python3 -m venv venv
```

**2.** Switch it on:

```bash
>>> source venv/bin/activate
```

Now there should be `(venv)` at the beginning of the line in your terminal.

> ⚠️ Every time you open a new terminal window, the venv is off again. Go to the `ml_for_artists` folder and run `source venv/bin/activate` again.

## Install the libraries

### Monday

```bash
>>> pip install torch numpy opencv-python
```

### Tuesday

```bash
>>> pip install torchvision diffusers datasets accelerate tensorboard tqdm matplotlib
```

If using `dataset_creation/prepare_faces.py`, also run
```bash
>>> pip install pillow-heif
```
