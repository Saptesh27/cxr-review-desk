# Run this project on another PC

Copy the project folder. Do not copy `.venv`, `data`, `cases.db`, or `checkpoints`. Those are created on the computer that runs the desk. Each machine gets its own virtual environment.

You need about 8 GB of free disk for the libraries and the three image models. The notes model adds about 13 GB and is downloaded only on a computer with an NVIDIA GPU. Training the fusion model needs network access once and can take roughly an hour, because three models run on 480 images.

No API key and no account are required.

Git is also needed. The localization model installs a small text encoder from Git the first time. Install Git for Windows from https://git-scm.com/download/win if `git --version` fails.

## 1. Install Python 3.11

Install Python 3.11 from https://www.python.org/downloads/release/python-3110/ and check **Add python.exe to PATH** on Windows.

This project is pinned to 3.11. A newer default, such as 3.14, can fail to install PyTorch.

Check:

```text
py -3.11 --version
```

On macOS or Linux:

```text
python3.11 --version
```

## 2. Open a terminal in the project folder

Windows example:

```text
cd C:\path\to\cxr-review-desk
```

## 3. Create the virtual environment

Windows:

```text
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

macOS or Linux:

```text
python3.11 -m venv .venv
source .venv/bin/activate
```

The prompt should now start with `(.venv)`.

## 4. Install the libraries

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Train the fusion model once

This is the training step. It does not retrain the four readers.

```text
python -m cxr.train_fusion
```

What you should see:

- A capped NIH subset saved in `data\nih_small` (400 train images, 80 test images).
- A line every few images while the three image agents run.
- `Saved checkpoints\fusion.joblib`
- A test macro-F1 printed in the terminal.

The same score is stored in `cases.db` and shows on the Training record page. If this file already exists, the desk still runs, using confidence weighting until you train.

You can stop here and skip training. The review desk works without it.

## 6. Start the desk

From the project folder, with the virtual environment activated. The two environment lines keep model progress bars from crashing on Windows.

Windows PowerShell:

```text
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn cxr.server:app --host 127.0.0.1 --port 7860
```

macOS or Linux:

```text
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
python -m uvicorn cxr.server:app --host 127.0.0.1 --port 7860
```

On Windows you can use the helper instead, after the virtual environment exists:

```text
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

## 7. Open the desk

In a browser, go to:

```text
http://127.0.0.1:7860
```

Leave the terminal window open while you use it.

## 8. Review one film

1. Choose **New study**.
2. Drop a frontal chest X-ray, or click the film area and pick a PNG or JPEG.
3. Type clinical notes if you have them. Notes can be left empty.
4. Click **Run review**.

The first review downloads model weights and can take several minutes. Later reviews are faster. The progress list shows which reader is working.

The result is either **Final diagnosis** or **Refer to radiologist**. It is saved under **History**.

## 9. Notes model

Agent 4 (CheXagent) loads only when `nvidia-smi` shows a GPU. On a computer without one, that card says **Not run** and the other three readers still produce the decision. That is expected.

## 10. Move the project again

Copy the source folder, then repeat steps 3 through 6 on the new computer. Do not copy `.venv`. Copying `checkpoints\fusion.joblib` is optional. If you copy it, you can skip step 5 and the desk will use the trained fusion model. Copying `data\nih_small` lets step 5 skip the download, but the three image models still have to be downloaded on the new computer the first time they run.

## If something fails

- `py -3.11` is not recognized: reinstall Python 3.11 and enable the py launcher, or use the full path to `python.exe`.
- `No module named cxr`: the terminal is not in the project folder, or the virtual environment is not activated.
- A model download stops: check the network and run the same command again. Finished files are cached.
- Port 7860 is already in use: change `7860` to `7861` in the uvicorn command and open that address instead.
- Training says an agent failed: read the sentence in the terminal. It names the reader and the reason. Fix that, then run `python -m cxr.train_fusion` again. Cached agent outputs in `data\nih_small\features.npz` are reused when the image count matches.

The explanation of the algorithms and the flow diagram is in `PROJECT.md`.
