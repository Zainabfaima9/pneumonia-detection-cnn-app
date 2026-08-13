# 🫁 AI-Assisted Pneumonia Triage from Chest X-Rays

> A prototype tool that flags chest X-rays showing signs consistent with pneumonia — built by a Medical Imaging Technology student to explore how AI can support imaging workflows, not replace radiologist interpretation.

**[🚀 Try the live app](#)** &nbsp;|&nbsp; **[📊 View results](#results)** &nbsp;|&nbsp; **[🧠 Model interpretability](#model-interpretability-grad-cam)**

---

## Why this project

I'm a Medical Imaging Technologist, not a radiologist — and that distinction is the actual starting point of this project. Technologists don't diagnose. We capture images and manage the workflow that decides how fast a patient's scan gets seen. In many settings, that workflow is the real bottleneck: a scan can be technically perfect and still sit in a queue if there aren't enough specialists to read it in time.

That's the gap I wanted to explore, hands-on: not "can AI diagnose pneumonia," but **can AI help prioritize which X-rays need a radiologist's eyes first?** So I taught myself Python from scratch, built and compared two deep learning models, and deployed one as a live web app that flags X-rays showing signs consistent with pneumonia for prioritized review (the CNN model-see live demo below).

This project is also a marker of the direction I want to take my career — staying rooted in imaging technology while building the technical fluency to design AI tools that actually fit into how imaging departments work, not just tools that score well on a test set.

---

## What it does

Upload a chest X-ray → the model flags it as **Normal** or **showing signs consistent with Pneumonia**, in real time, through a live web app — framed as a triage aid, not a diagnosis.

| | |
|---|---|
| 🖼️ **Input** | Chest X-ray image (JPEG/PNG) |
| 🧠 **Models** | Custom CNN \| VGG16 (Transfer Learning) |
| 👁️ **Interpretability** | Grad-CAM heatmaps — shows *where* the model is looking |
| ⚡ **Output** | Instant flag + visual explanation |
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

![Training Accuracy 1](Screenshot/training_accuracy.png)
![Training Accuracy 2](Screenshot/training_accuracy_1.png)
![Training Accuracy 3](Screenshot/training_accuracy_2.png)
![Training Accuracy 4](Screenshot/training_accuracy_3.png)
![Training Accuracy 5](Screenshot/training_accuracy_4.png)
![Training Accuracy 6](Screenshot/training_accuracy_5.png)

![Pneumonia Detection Result](Screenshot/pneumonia_result.png)

![Normal Result](Screenshot/normal_result.png)

---

## Model Interpretability (Grad-CAM)

A high accuracy score means nothing to a clinician if they can't see *why* the model made a call — and for a tool meant to support a real imaging workflow, that trust gap is the whole problem, not a footnote. So I applied **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize which regions of each X-ray most influenced the model's prediction.

![Grad-CAM Heatmap](Screenshot/gradcam_result.png)

On the pneumonia-flagged X-ray above, the heatmap concentrates on the lower lung fields — the anatomical region where pneumonia-related opacity typically appears — rather than on bone, soft tissue, or image edges. That's not proof the model is clinically reliable, but it is evidence it's attending to plausible regions rather than shortcuts in the image, which is the minimum a technologist or radiologist would need to see before trusting a flag like this at all.

---

## 🚀 Live Demo

**[Try it yourself → https://pneumonia-detection-cnn-app-a4j2vpt7bpfvfqzupxdjwn.streamlit.app/](#)**

Upload any chest X-ray and see the model's prediction instantly *(this demo uses the CNN model; VGG16 results are shown above for comparison)*. The trained model is hosted on Hugging Face Hub and loaded dynamically by the app, due to file size constraints on GitHub.

---

## From Prototype to Practice

Building the model was the easy part. Turning it into something an imaging department could actually trust and use is a different problem entirely — and it's the problem I'm more interested in than the model itself:

- **Workflow fit:** A triage tool only helps if it plugs into how technologists already prioritize studies — not if it adds a second system to check.
- **Clinical validation:** This model has seen one small, single-source dataset. Real deployment would require testing across diverse patient populations, scanner types, and clinical sites.
- **Regulatory pathway:** Any AI tool touching patient care — even a triage aid, not a diagnostic one — needs a clear route through medical device regulation (e.g., FDA, CE marking) before it can be used outside a research setting.
- **Trust and adoption:** Explainability tools like Grad-CAM are a start, but adoption ultimately depends on technologists and radiologists being part of building the tool, not just receiving it.

This is the layer I want to spend my career working in: not just building models, but understanding how a promising prototype like this one actually becomes a tool a hospital would adopt.

---

## Limitations

- Trained on a relatively small dataset (~5,800 images) from a single source
- No demographic diversity metadata available
- Not validated against real clinical outcomes
- Flags signs *consistent with* pneumonia — it does not diagnose severity, type (bacterial vs. viral), or any other condition
- Built for education and demonstration — not for actual clinical use

---

## A note on process

This project started from a publicly available Kaggle tutorial notebook, which I used to learn the core workflow of building and training a CNN for image classification. From there, I adjusted the training setup, evaluated both models independently, added Grad-CAM interpretability, wrote my own analysis, and deployed the final model as a live app. I think being upfront about that starting point — rather than pretending it began from a blank file — is part of doing this honestly.

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `Kaggle Notebooks (GPU)` · `Streamlit` · `Hugging Face Hub (model hosting)` · `Grad-CAM` · `GitHub`

---

## Author

**Zainab Fatima**
BS Medical Imaging Technology
