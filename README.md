# 🫁 AI-Based Pneumonia Detection from Chest X-Rays

> An AI-assisted screening tool that classifies chest X-rays as *Normal* or *Pneumonia*, built to explore how deep learning can support faster, more accessible diagnosis — especially where radiologist access is limited.

**[🚀 Try the live app](#)** &nbsp;|&nbsp; **[📊 View results](#results)** &nbsp;|&nbsp; **[🧠 Read the methodology](#methodology)**

---

## Why this project

Diagnostic imaging is only as fast as the specialists available to read it — and in many parts of the world, that's the real bottleneck, not the equipment. As a Medical Imaging Technology student, I wanted to understand, hands-on, whether AI could meaningfully close that gap: not replacing a radiologist's judgment, but giving frontline healthcare workers a fast, first-pass signal when one isn't immediately available.

This project is also a personal marker of the direction I want to take my career — staying rooted in clinical imaging while building the technical fluency to eventually design AI tools that are actually usable in the settings that need them most.

---

## What it does

Upload a chest X-ray → the model classifies it as **Normal** or **Pneumonia** in real time, through a live web app.

| | |
|---|---|
| 🖼️ **Input** | Chest X-ray image (JPEG/PNG) |
| 🧠 **Models** | Custom CNN \| VGG16 (Transfer Learning) |
| ⚡ **Output** | Instant classification |
| 🌐 **Deployment** | Streamlit Cloud, live and public |

---

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** ~5,800 labeled chest X-ray images
- **Classes:** Normal, Pneumonia
- **Preprocessing:** Resizing, normalization, train/test split

---

## Methodology

Two models were built and compared, to see how a model trained from scratch stacks up against transfer learning on a limited medical imaging dataset:

1. **Custom CNN** — designed and trained from the ground up
2. **VGG16 (Transfer Learning)** — a pretrained network fine-tuned on the chest X-ray dataset

Both were trained for 15 epochs on an 80/20 train-test split, then evaluated on a held-out test set.

---

## Results

| Model | Test Accuracy |
|---|---|
| Custom CNN | **89%** |
| VGG16 (Transfer Learning) | **92%** ✅ |

VGG16 came out ahead — and that gap is the interesting part, not just the numbers. It's a clear, hands-on demonstration of *why* transfer learning matters for medical imaging: the pretrained network arrived already knowing how to recognize general visual patterns, so it needed far less data to specialize in X-rays than a model starting from zero.

![Training Accuracy 1](screenshots/training_accuracy.png)
![Training Accuracy 2](screenshots/training_accuracy_1.png)
![Training Accuracy 3](screenshots/training_accuracy_2.png)
![Training Accuracy 4](screenshots/training_accuracy_3.png)
![Training Accuracy 5](screenshots/training_accuracy_4.png)
![Training Accuracy 6](screenshots/training_accuracy_5.png)

![Pneumonia Detection Result](screenshots/pneumonia_result.png)

![Normal reusult](screenshots/normal_result.png)

---

## 🚀 Live Demo

**[Try it yourself → \[https://pneumonia-detection-cnn-app-a4j2vpt7bpfvfqzupxdjwn.streamlit.app/]](#)**

Upload any chest X-ray and see the model's prediction instantly — no setup required.

---

## Clinical Interpretation

This model performs well on its test set, but it's trained on a modest, single-source public dataset and hasn't been clinically validated — so it's not a diagnostic tool. What it *is*: a working proof of concept for how AI-assisted screening could one day give healthcare workers a fast, useful signal in settings where specialist review is delayed or unavailable. Getting from here to real-world clinical use would take far larger and more diverse data, rigorous validation, and regulatory approval.

---

## Limitations

- Trained on a relatively small dataset (~5,800 images) from a single source
- No demographic diversity metadata available
- Not validated against real clinical outcomes
- Built for education and demonstration — not for actual diagnosis

---

## A note on process

This project started from a publicly available Kaggle tutorial notebook, which I used to learn the core workflow of building and training a CNN for image classification. From there, I adjusted the training setup, evaluated both models independently, wrote my own analysis, and deployed the final model as a live app. I think being upfront about that starting point — rather than pretending it began from a blank file — is part of doing this honestly.

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `Kaggle Notebooks (GPU)` · `Streamlit` · `GitHub`

---

## Author

**Zainab Fatima**
BS Medical Imaging Technology
