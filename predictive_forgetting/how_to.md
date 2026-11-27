<!-- Open this file on github: https://github.com/juliaschicker/ml_for_artists_hs2026/blob/main/predictive_forgetting/how_to.md -->
# Predictive forgetting

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.
> - Grey boxes **without** `>>>` show how a line in a Python file should look.
>   Change that line in Sublime, then save the file.

---

## Preparation of the files

**1.** Split your video into single frames with ffmpeg (see `ffmpeg.md`). Make sure the frames are saved in the `predictive_forgetting` folder.

## Training

**2.** Set the training folder in `train_predictive_forgetting.py`. Go to **line 51**:

```python
FRAMES_DIR = "training_frames"
```

Change `training_frames` to the name of the folder with your training footage. Ideally, the folder has a few hundred files.

**3.** Set the name of the model on **line 56**, e.g.:

```python
CHECKPOINT_OUT = "predictive_forgetting_video_1.pt"
```

Then you know which footage you trained it on.

> ⚠️ Change the name every time. Otherwise, your old model is overwritten!

**4.** Save and close the file.

**5.** In your terminal, make sure you are in the `predictive_forgetting` folder. Check the text right before your cursor.

If you are not there, type `cd`, a space, and then press the **Tab** key. Pick the path to your `predictive_forgetting` folder:

```bash
>>> cd 
```

If you need to go one folder up, type this and press Enter:

```bash
>>> cd ..
```

**6.** In your terminal, check that the venv is switched on: there should be `(venv)` at the beginning of the line. If so, run:

```bash
>>> python train_predictive_forgetting.py
```

**7.** If the epochs pass very slowly, call me.

**8.** Voilà, you have a model ready!

## Output

**9.** Open `wake_sleep_consolidate.py`.

**10.** On **line 44**, change the model name to the name you gave your model before:

```python
CHECKPOINT = "your-model-name.pt"
```

**11.** On **line 46**, set the name of the folder with the images you want to process (e.g. the ones you trained on):

```python
FRAMES_DIR = "your-frames-folder"
```

**12.** On **line 48**, set a folder for your output images. It is created automatically.

```python
OUTPUT_DIR = "your-output-folder"
```

> ⚠️ Use a new folder name every time. Otherwise, your old images are overwritten!
> Use names that make sense and maybe also tell you which model you used. Otherwise, this gets confusing very quickly.

**13.** Go to your terminal and run:

```bash
>>> python wake_sleep_consolidate.py
```

**14.** To turn your images back into a video, change `folder` (twice) in the line below to the name of your output folder. Then copy it and run it in your terminal:

```bash
>>> ffmpeg -framerate 25 -pattern_type glob -i 'folder/*.jpg' -c:v libx264 -pix_fmt yuv420p folder/out.mp4
```

**15.** You'll find your video `out.mp4` in your output folder!

---

## Alternatives to try out (images)

1. Apply the model to a different video than the one it was trained on. In `wake_sleep_consolidate.py`, change `FRAMES_DIR` to the frames folder of another video. Also change `OUTPUT_DIR`!
2. Use the two videos of "opposites" you shot: train the model on one, apply it to both videos, and compare the results.
3. Make a training set with frames from different videos.
4. Come up with your own combinations of images to train on and images to compress.

## Other things to try out (parameters)

> ⚠️ Change the name in `CHECKPOINT_OUT` (line 56) every time. Otherwise, your other model is overwritten!

1. `train_predictive_forgetting.py`, **line 72**: `SUCCESSOR_WINDOW`
   Make it smaller or larger. It defines what the "successor state" is.
2. `train_predictive_forgetting.py`, **line 79**: `EPOCHS`
   Smaller makes the model coarser, larger means it's "trained better". The training stops automatically when it doesn't improve anymore.
3. Search for the other `TODO`s in `train_predictive_forgetting.py` and play around with them.
4. Search for the other `TODO`s in `wake_sleep_consolidate.py` and play around with them.
