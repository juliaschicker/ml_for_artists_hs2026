# How to move around in the terminal

> **How to read this file**
>
> - Grey boxes that start with `>>>` are **commands for the terminal**.
>   Copy only the part **after** `>>>`. **Do not copy the `>>>` itself!** Then press Enter.

---

## Go one folder deeper

Like double-clicking a folder in the Finder: from the folder you are in, you go into one of its folders.

```bash
>>> cd name_of_the_folder
```

## See which folders you can go to

Type `cd`, a space, and then press the **Tab** key. This shows a list of all the folders inside the current one.

```bash
>>> cd 
```

## Autocomplete a folder name

Type `cd`, a space, the first few letters of the folder name, and then press **Tab**. The terminal completes the name for you.

```bash
>>> cd my-fo
```

## Go one folder up

```bash
>>> cd ..
```

## See all files and folders in the current folder

```bash
>>> ls
```
