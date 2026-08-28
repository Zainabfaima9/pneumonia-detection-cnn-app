import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# =========================================================
# ⚠️ IMPORTANT — VERIFY BEFORE DEPLOYING ⚠️
# The preprocessing below (image size, grayscale vs RGB,
# normalization) MUST match exactly what you used when you
# trained each model in your notebook. If it doesn't match,
# predictions will be wrong even though the app runs fine.
# =========================================================

CNN_INPUT_SIZE = (128, 128)     # grayscale, matches your existing CNN app
VGG16_INPUT_SIZE = (256, 256)   # confirmed from deployment error: model expects 256x256x3

# =========================================================
# 📁 SAMPLE IMAGES — you need to add these yourself
# Create a folder called `sample_images/` in your repo (next to
# app.py) and add 4 chest X-rays copied from your own Kaggle
# test set (the same dataset you trained on):
#   sample_images/normal_1.jpg
#   sample_images/normal_2.jpg
#   sample_images/pneumonia_1.jpg
#   sample_images/pneumonia_2.jpg
# Using images from your own training/test data keeps this fully
# authentic and avoids any licensing question with outside images.
# =========================================================
SAMPLE_IMAGES = {
    "Normal Example 1": "sample_images/normal_1.jpg",
    "Normal Example 2": "sample_images/normal_2.jpg",
    "Pneumonia Example 1": "sample_images/pneumonia_1.jpg",
    "Pneumonia Example 2": "sample_images/pneumonia_2.jpg",
}

st.set_page_config(
    page_title="Pneumonia Triage AI",
    page_icon="🫁",
    layout="wide"
)

# ---------------------------------------------------------
# Model loading (cached so it only downloads/loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    path = hf_hub_download(
        repo_id="zainabfatima9/pnemonia-cnn-model",
        filename="cnn_model.h5"
    )
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_vgg16_model():
    path = hf_hub_download(
        repo_id="zainabfatima9/pneumonia-vgg16-model",
        filename="vgg16_model.keras"
    )
    return tf.keras.models.load_model(path)


def preprocess_for_cnn(pil_image):
    img = pil_image.convert("L").resize(CNN_INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_for_vgg16(pil_image):
    img = pil_image.convert("RGB").resize(VGG16_INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, size):
    import matplotlib.cm as cm
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(size)
    heatmap_arr = np.array(heatmap_img)
    colored = cm.jet(heatmap_arr / 255.0)[:, :, :3]
    colored = np.uint8(colored * 255)
    base = np.array(pil_image.convert("RGB").resize(size))
    blended = np.uint8(base * 0.55 + colored * 0.45)
    return Image.fromarray(blended)


def run_full_analysis(image, cnn_model, vgg_model):
    """Runs both models + Grad-CAM in one go and returns everything needed to display."""
    cnn_input = preprocess_for_cnn(image)
    vgg_input = preprocess_for_vgg16(image)

    cnn_score = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
    vgg_score = float(vgg_model.predict(vgg_input, verbose=0)[0][0])

    last_conv = find_last_conv_layer(cnn_model)
    heatmap = make_gradcam_heatmap(cnn_input, cnn_model, last_conv) if last_conv else None
    overlay = overlay_heatmap(image, heatmap, size=(320, 320)) if heatmap is not None else None

    return cnn_score, vgg_score, overlay


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title("🫁 Pneumonia Triage AI")

st.sidebar.markdown(
    """
    **About this project**

    Built by **Zainab Fatima**, a Medical Imaging Technology student, to explore
    a narrower question than *"can AI diagnose pneumonia?"* — namely, **can AI
    help prioritize which chest X-rays need a radiologist's eyes first?**

    A prototype triage aid, not a diagnostic tool.
    """
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Diagnose an X-ray", "Model Comparison", "Triage Queue Simulator", "Future Vision"]
)

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "Home":
    st.title("🫁 AI-Assisted Pneumonia Triage from Chest X-Rays")
    st.caption("A prototype triage tool by Zainab Fatima — Medical Imaging Technology")

    st.markdown(
        """
        ### About Me & This Project

        I'm a Medical Imaging Technology student — not a radiologist — and that
        distinction is the actual starting point of this project. Technologists
        don't diagnose. We capture images and manage the workflow that decides how
        fast a patient's scan gets seen. In many settings, that workflow is the
        real bottleneck: a scan can be technically perfect and still sit in a queue
        if there aren't enough specialists to read it in time.

        That's the gap I wanted to explore hands-on: not *"can AI diagnose
        pneumonia?"*, but **can AI help prioritize which X-rays need a
        radiologist's eyes first?** So I taught myself deep learning from scratch,
        built and compared two models (a custom CNN and a VGG16 transfer-learning
        model), added Grad-CAM interpretability so predictions aren't a black box,
        and deployed both as this live triage tool.
        """
    )

    st.subheader("Why this matters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Global cases per year", "~450 million")
        st.caption("Roughly 7% of the world's population is affected by pneumonia every year.")
    with col2:
        st.metric("Global deaths per year", "~4 million")
        st.caption("Pneumonia is a leading infectious cause of death across all age groups worldwide.")
    with col3:
        st.metric("Child deaths (under 5), 2021", "~500,000+")
        st.caption("Pneumonia remains the single largest infectious killer of children under five globally.")

    st.markdown(
        """
        **Pakistan carries a disproportionate share of this burden.** Multiple global
        health studies (WHO / UNICEF / Global Burden of Disease estimates) consistently
        place Pakistan among the handful of countries — alongside India, Nigeria, and
        the DR Congo — with the highest number of under-five pneumonia deaths in the
        world, with annual estimates in the tens of thousands of children. Pakistan has
        also historically ranked among the countries with the highest number of
        childhood pneumonia *cases* per year, reflecting both disease burden and gaps
        in early diagnosis and access to timely imaging review.

        *(Figures are drawn from publicly available WHO, UNICEF, and Global Burden of
        Disease reporting and are approximate — exact numbers vary by year and source.)*
        """
    )

    st.subheader("Why triage — not just detection")
    st.markdown(
        """
        Pneumonia can move from mild to life-threatening in a matter of hours,
        especially in young children and elderly patients. In many hospitals —
        particularly in low-resource settings — a chest X-ray can sit in a queue
        for hours before a radiologist reviews it, simply because there aren't
        enough specialists to read every scan the moment it's taken.

        **Early diagnosis directly changes outcomes**: antibiotics started sooner,
        oxygen support started sooner, and fewer complications. This project
        doesn't try to replace the radiologist's diagnosis — it tries to answer a
        narrower, more practical question: **can AI help make sure the most urgent
        X-rays are seen first?**
        """
    )

    st.info(
        "Use the sidebar to try the tool yourself, compare the two models, "
        "simulate a real triage queue, or read about where this could go next."
    )

# ---------------------------------------------------------
# DIAGNOSE AN X-RAY — upload + dual model + Grad-CAM, all together
# ---------------------------------------------------------
elif page == "Diagnose an X-ray":
    st.title("Diagnose an X-ray")
    st.write(
        "Upload your own chest X-ray, or pick a sample below. You'll immediately "
        "see both models' predictions **and** the Grad-CAM heatmap together."
    )

    source = st.radio("Choose an image source", ["Upload my own", "Use a sample X-ray"], horizontal=True)

    image = None

    if source == "Upload my own":
        uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        available = {name: path for name, path in SAMPLE_IMAGES.items() if os.path.exists(path)}
        if not available:
            st.warning(
                "No sample images found yet. Add a `sample_images/` folder with a few "
                "chest X-rays from your Kaggle test set to enable this option."
            )
        else:
            choice = st.selectbox("Pick a sample", list(available.keys()))
            image = Image.open(available[choice])

    if image is not None:
        col_img, col_results = st.columns([1, 2])

        with col_img:
            st.image(image, caption="X-ray being analyzed", width="stretch")

        with st.spinner("Loading models and analyzing..."):
            cnn_model = load_cnn_model()
            vgg_model = load_vgg16_model()
            cnn_score, vgg_score, gradcam_overlay = run_full_analysis(image, cnn_model, vgg_model)

        with col_results:
            st.subheader("Predictions")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Custom CNN**")
                st.metric("Confidence (Pneumonia)", f"{cnn_score * 100:.1f}%")
                st.progress(min(max(cnn_score, 0.0), 1.0))
                st.error("🦠 Signs consistent with Pneumonia") if cnn_score >= 0.5 else st.success("✅ Normal")
            with c2:
                st.markdown("**VGG16 (Transfer Learning)**")
                st.metric("Confidence (Pneumonia)", f"{vgg_score * 100:.1f}%")
                st.progress(min(max(vgg_score, 0.0), 1.0))
                st.error("🦠 Signs consistent with Pneumonia") if vgg_score >= 0.5 else st.success("✅ Normal")

            agree = (cnn_score >= 0.5) == (vgg_score >= 0.5)
            if agree:
                st.info("✅ Both models agree on this prediction.")
            else:
                st.warning(
                    "⚠️ The two models disagree — exactly the kind of case that "
                    "would benefit most from a radiologist's review."
                )

        st.markdown("---")
        st.subheader("Grad-CAM: Where the model is looking")
        if gradcam_overlay is not None:
            g1, g2 = st.columns(2)
            with g1:
                st.image(image.resize((320, 320)), caption="Original X-ray")
            with g2:
                st.image(gradcam_overlay, caption="Grad-CAM Heatmap (Custom CNN)")
            st.caption(
                "Warmer regions (red/yellow) indicate areas the CNN weighted most "
                "heavily in its prediction. Ideally these fall over the lung fields, "
                "not on bone, edges, or background artifacts — a basic sanity check "
                "before trusting a flag like this at all."
            )
        else:
            st.warning("Could not generate a Grad-CAM heatmap for this model.")

# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write(
        "Two models were built and compared, to see how a model trained from "
        "scratch stacks up against transfer learning on a limited medical imaging dataset."
    )

    st.table({
        "Model": ["Custom CNN", "VGG16 (Transfer Learning)"],
        "Test Accuracy": ["89%", "92%"],
        "Input": ["128x128 grayscale", "224x224 RGB"],
        "Approach": ["Trained from scratch", "Pretrained on ImageNet, fine-tuned"]
    })

    st.markdown(
        """
        **VGG16 is the stronger model overall, and the recommended one for real use.**
        It scored higher on the held-out test set (92% vs. 89%), and that gap reflects
        something deeper than a couple of percentage points: VGG16 arrived already
        knowing how to recognize general visual patterns — edges, textures, shapes —
        from being pretrained on millions of images. It only needed to *specialize*
        that existing knowledge for chest X-rays, rather than learn to see from zero
        the way the custom CNN had to on a comparatively small medical dataset.

        In a real deployment, VGG16 would be the primary model driving triage
        decisions, with the custom CNN kept as a lightweight baseline for comparison
        and for the Grad-CAM interpretability view. This is also the direction most
        production medical-imaging AI takes in practice — transfer learning on top
        of large pretrained vision backbones, rather than training small models from
        scratch on limited clinical data.
        """
    )

# ---------------------------------------------------------
# TRIAGE QUEUE SIMULATOR
# ---------------------------------------------------------
elif page == "Triage Queue Simulator":
    st.title("Triage Queue Simulator")
    st.write(
        "This is the actual workflow idea behind the project: instead of X-rays "
        "waiting in the order they arrived, an AI-prioritized queue reorders them "
        "by urgency. Below is a demo queue built from sample X-rays — or upload "
        "your own batch."
    )

    use_demo = st.checkbox("Use built-in demo queue (sample X-rays)", value=True)

    files_to_score = []  # list of (name, PIL image)

    if use_demo:
        for name, path in SAMPLE_IMAGES.items():
            if os.path.exists(path):
                files_to_score.append((name, Image.open(path)))
        if not files_to_score:
            st.warning(
                "No sample images found yet. Add a `sample_images/` folder "
                "(see the comment at the top of app.py) to enable the demo queue."
            )

    uploaded_files = st.file_uploader(
        "Or upload your own batch of Chest X-rays",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    for f in uploaded_files:
        files_to_score.append((f.name, Image.open(f)))

    if files_to_score:
        cnn_model = load_cnn_model()
        results = []

        with st.spinner(f"Scoring {len(files_to_score)} X-rays..."):
            for name, img in files_to_score:
                score = float(cnn_model.predict(preprocess_for_cnn(img), verbose=0)[0][0])
                results.append((name, img, score))

        results.sort(key=lambda x: x[2], reverse=True)

        st.subheader("Prioritized Reading Queue")
        for rank, (name, img, score) in enumerate(results, start=1):
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                st.image(img, width=90)
            with col2:
                st.write(f"**#{rank} — {name}**")
                st.progress(min(max(score, 0.0), 1.0))
            with col3:
                if score >= 0.5:
                    st.error(f"🦠 {score*100:.1f}% — flagged for priority review")
                else:
                    st.success(f"✅ {score*100:.1f}% — normal")

        st.caption(
            "In a real workflow, this reordering — not a diagnosis — is the "
            "actual value: getting the most urgent-looking scans in front of a "
            "radiologist sooner, without changing who ultimately makes the call."
        )

# ---------------------------------------------------------
# FUTURE VISION
# ---------------------------------------------------------
elif page == "Future Vision":
    st.title("Future Vision")
    st.write(
        "This prototype is a starting point, not a finished product. Here's how it "
        "could realistically evolve toward something an imaging department could use:"
    )

    st.markdown(
        """
        - **Workflow integration:** Connect directly to a hospital's PACS/RIS system
          so flagged X-rays are automatically bumped up in the radiologist's reading
          list, instead of requiring a separate app.
        - **Multi-class detection:** Extend beyond Normal vs. Pneumonia to
          distinguish bacterial vs. viral pneumonia, and flag other common findings
          (e.g. pleural effusion, TB-suggestive patterns), since these change
          treatment decisions.
        - **Severity scoring:** Move from a binary flag to a severity score, so
          radiologists see not just "pneumonia present" but a rough sense of how
          urgent the case is.
        - **Larger, multi-site validation:** Test and retrain on X-rays from
          multiple hospitals, scanner types, and patient populations — this
          prototype has only seen one relatively small, single-source dataset.
        - **Edge deployment for low-resource clinics:** Package a lightweight
          version that can run on modest hardware in clinics without a
          radiologist on-site, where triage delay is often the biggest problem.
        - **Regulatory pathway:** Any tool that touches real patient care —
          even a triage aid, not a diagnostic one — needs to go through proper
          medical device regulation (e.g. FDA, CE marking) before real-world use.
        - **Combined imaging dashboard:** Pair this with other imaging AI tools
          (such as a CT image-quality/dose-optimization tool) into a single
          department-wide triage and quality dashboard, rather than a standalone app.
        """
    )

    st.info(
        "The technical model is the easy part. The harder, more interesting "
        "problem — and the one worth building a career around — is making a tool "
        "like this actually fit into how imaging departments already work."
    )
