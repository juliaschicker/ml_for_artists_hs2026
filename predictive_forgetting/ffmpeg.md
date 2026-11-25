# Splitting a video into frames with ffmpeg

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.

---

**1.** In your terminal, go to the `predictive_forgetting` folder. Your frames will be saved there.

**2.** Make one output folder per video, e.g. `video-1`. This is where the single frames are stored:

```bash
>>> mkdir video-1
```

**3.** In the line below, change `video.mp4` to the name of your video file. Then run it:

```bash
>>> ffmpeg -i video.mp4 video-1/frame-%04d.jpg
```

> ⚠️ The video file name must not contain any spaces.

## What the parts mean

| Part | Meaning |
|------|---------|
| `ffmpeg` | calls the ffmpeg programme, which does everything that comes after it |
| `-i video.mp4` | `-i` stands for "input": `video.mp4` is the video the programme works with |
| `video-1/` | the folder it saves the frames to |
| `frame-%04d.jpg` | the file names it uses. `%04d` means four digits with leading zeros: `frame-0001.jpg`, `frame-0002.jpg`, etc. |
| `.jpg` | saves the frames as JPGs. You could also use `.png`, but it takes much more space, and for this experiment there's no reason to do it. |
