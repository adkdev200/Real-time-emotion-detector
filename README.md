# Mood Analysis — Real-time Emotion Detection with Vision Transformer

I built this because I was curious about one thing: *can a model actually tell what I'm feeling just by looking at my face through a webcam?* Turns out — yeah, it kind of can. And it's surprisingly fun to watch it work in real time.

---

## What is this?

This is a real-time facial emotion recognition system. You run it, point your webcam at your face, and it tells you what emotion it thinks you're expressing — live, on screen.

It detects **6 emotions**:
- 😮 Ahegao
- 😠 Angry
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

Under the hood it uses a **Vision Transformer (ViT-B/16)** fine-tuned on an emotion recognition dataset, with **MediaPipe** doing the heavy lifting of finding your face in the frame before passing the crop to the model.

---

## How it works

The pipeline is pretty straightforward:

1. **Webcam captures frames** continuously via OpenCV
2. **MediaPipe Face Detection** finds every face in the frame and draws a bounding box around it
3. The face crop gets **preprocessed** (resized, normalized) to match what the ViT model expects
4. The **trained ViT model** predicts the emotion class
5. The predicted label gets **drawn on screen** next to the bounding box — in real time

The model was trained using the [Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset) from Kaggle.

---

## Project structure

```
mood_analysis/
│
├── train.py          # Model architecture + training loop
├── train.ipynb       # Notebook version of the training process
├── inference.py      # Real-time webcam inference script
├── requirements.txt  # All dependencies
└── README.md
```

The trained model weights (`emotion_transformer.pt`) are not included in this repo because the file is ~330MB. You'll need to train it yourself or reach out if you want the weights directly.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/mood_analysis.git
cd mood_analysis
```

### 2. Set up a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure you have a CUDA-capable GPU if you want fast inference. It'll fall back to CPU but will be noticeably slower.

### 4. Train the model (or skip if you have the weights)

You'll need a Kaggle account and the `kagglehub` package (already in requirements) to download the dataset automatically.

```bash
python train.py
```

This saves the model as `emotion_transformer.pt` in the project root.

### 5. Run real-time inference

```bash
python inference.py
```

A window will open showing your webcam feed. The model will start predicting emotions on any face it detects. Press **`q`** to quit.

---

## Tech stack

| Tool | Purpose |
|---|---|
| PyTorch | Model training & inference |
| torchvision ViT-B/16 | Pretrained Vision Transformer backbone |
| MediaPipe | Face detection |
| OpenCV | Webcam capture & frame rendering |
| kagglehub | Dataset download |
| scikit-learn | Label encoding during training |

---

## Notes & things I'd improve

- The model was trained for only a couple of epochs so there's definitely room to push the accuracy higher with longer training or data augmentation.
- The `Ahegao` class in the dataset is a bit of an odd one — it occasionally fires when the model gets confused, which is... amusing.
- Multi-face support already works since MediaPipe detects all faces in the frame.
- I'd love to add a confidence threshold so that low-confidence predictions just show "Unknown" instead of guessing.

---

## Requirements

Python 3.10+ is recommended. Full dependency list is in `requirements.txt`. The key ones are:

- `torch >= 2.0`
- `torchvision`
- `mediapipe`
- `opencv-python`
- `Pillow`

---

If you try it out and something breaks or you get a weird result, feel free to open an issue. Always happy to take a look.
