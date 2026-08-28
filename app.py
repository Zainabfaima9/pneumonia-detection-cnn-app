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
# trained each model in your notebook. If VGG16 was trained
# on 224x224 RGB images normalized a particular way, make sure
# VGG16_INPUT_SIZE / vgg16 preprocessing below matches that.
# If it doesn't match, VGG16 predictions will be wrong even
# though the app runs without errors.
# =========================================================

CNN_INPUT_SIZE = (128, 128)     # grayscale, matches your existing CNN app
VGG16_INPUT_SIZE = (224, 224)   # standard VGG16 input — CHANGE if you trained differently

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
    arr = np.expand_dims(arr, axis=-1)   # channel dim
    arr = np.expand_dims(arr, axis=0)    # batch dim
    return arr


def preprocess_for_vgg16(pil_image):
    img = pil_image.convert("RGB").resize(VGG16_INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def find_last_conv_layer(model):
    """Walk backwards through a model's layers to find the last Conv2D layer.
    Works for simple Sequential/Functional CNNs. For nested models (like a
    VGG16 base wrapped inside a bigger model) this only searches the top level."""
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


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title("🫁 Pneumonia Triage AI")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Try It Yourself", "Model Comparison", "Grad-CAM Explorer", "Triage Queue Simulator"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "A prototype triage aid, not a diagnostic tool. "
    "Built by a Medical Imaging Technology student."
)

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "Home":
    st.title("AI-Assisted Pneumonia Triage from Chest X-Rays")
    st.markdown(
        """
        Technologists don't diagnose — we capture images and manage the workflow
        that decides how fast a patient's scan gets seen. In many settings, that
        workflow is the real bottleneck: a scan can be technically perfect and
        still sit in a queue if there aren't enough specialists to read it in time.

        This tool explores a narrower question than *"can AI diagnose pneumonia?"* —
        it asks **can AI help prioritize which X-rays need a radiologist's eyes first?**

        Use the sidebar to:
        - **Try It Yourself** — upload an X-ray and see both models' predictions side by side
        - **Model Comparison** — see how the custom CNN stacks up against VGG16 transfer learning
        - **Grad-CAM Explorer** — see *where* the model is looking on your X-ray
        - **Triage Queue Simulator** — upload several X-rays and watch the AI reorder them by urgency
        """
    )

# ---------------------------------------------------------
# TRY IT YOURSELF — dual model comparison
# ---------------------------------------------------------
elif page == "Try It Yourself":
    st.title("Try It Yourself")
    st.write("Upload a chest X-ray to see how the Custom CNN and VGG16 models each score it.")

    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", width=350)

        with st.spinner("Loading models and predicting..."):
            cnn_model = load_cnn_model()
            vgg_model = load_vgg16_model()

            cnn_input = preprocess_for_cnn(image)
            vgg_input = preprocess_for_vgg16(image)

            cnn_score = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
            vgg_score = float(vgg_model.predict(vgg_input, verbose=0)[0][0])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Custom CNN")
            st.metric("Confidence (Pneumonia)", f"{cnn_score * 100:.1f}%")
            st.progress(min(max(cnn_score, 0.0), 1.0))
            if cnn_score >= 0.5:
                st.error("🦠 Signs consistent with Pneumonia")
            else:
                st.success("✅ Normal")

        with col2:
            st.subheader("VGG16 (Transfer Learning)")
            st.metric("Confidence (Pneumonia)", f"{vgg_score * 100:.1f}%")
            st.progress(min(max(vgg_score, 0.0), 1.0))
            if vgg_score >= 0.5:
                st.error("🦠 Signs consistent with Pneumonia")
            else:
                st.success("✅ Normal")

        st.markdown("---")
        agree = (cnn_score >= 0.5) == (vgg_score >= 0.5)
        if agree:
            st.info("✅ Both models agree on this prediction.")
        else:
            st.warning(
                "⚠️ The two models disagree — this is exactly the kind of case "
                "that would benefit most from a radiologist's review."
            )

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
        VGG16 came out ahead — and that gap is the interesting part, not just the
        numbers. It's a clear, hands-on demonstration of *why* transfer learning
        matters for medical imaging: the pretrained network arrived already knowing
        how to recognize general visual patterns, so it needed far less data to
        specialize in X-rays than a model starting from zero.
        """
    )

# ---------------------------------------------------------
# GRAD-CAM EXPLORER
# ---------------------------------------------------------
elif page == "Grad-CAM Explorer":
    st.title("Grad-CAM Explorer")
    st.write(
        "A high accuracy score means nothing to a clinician if they can't see "
        "*why* the model made a call. Upload an X-ray to see which regions the "
        "**Custom CNN** focused on most."
    )
    st.caption("Note: Grad-CAM is currently shown for the Custom CNN model.")

    uploaded_file = st.file_uploader(
        "Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"], key="gradcam_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        cnn_model = load_cnn_model()
        cnn_input = preprocess_for_cnn(image)

        last_conv = find_last_conv_layer(cnn_model)

        if last_conv is None:
            st.error("Could not find a Conv2D layer in this model to visualize.")
        else:
            with st.spinner("Generating Grad-CAM heatmap..."):
                heatmap = make_gradcam_heatmap(cnn_input, cnn_model, last_conv)
                overlay = overlay_heatmap(image, heatmap, size=(300, 300))

            col1, col2 = st.columns(2)
            with col1:
                st.image(image.resize((300, 300)), caption="Original X-ray")
            with col2:
                st.image(overlay, caption="Grad-CAM Heatmap")

            st.caption(
                "Warmer regions (red/yellow) indicate areas the model weighted most "
                "heavily in its prediction. Ideally these should fall over lung fields, "
                "not on bone, edges, or background artifacts."
            )

# ---------------------------------------------------------
# TRIAGE QUEUE SIMULATOR
# ---------------------------------------------------------
elif page == "Triage Queue Simulator":
    st.title("Triage Queue Simulator")
    st.write(
        "This is the actual workflow idea behind the project: instead of X-rays "
        "waiting in the order they arrived, upload several at once and see how "
        "an AI-prioritized queue would reorder them by urgency."
    )

    uploaded_files = st.file_uploader(
        "Upload multiple Chest X-rays",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        cnn_model = load_cnn_model()
        results = []

        with st.spinner(f"Scoring {len(uploaded_files)} X-rays..."):
            for f in uploaded_files:
                img = Image.open(f)
                score = float(cnn_model.predict(preprocess_for_cnn(img), verbose=0)[0][0])
                results.append((f.name, img, score))

        # Sort by descending pneumonia confidence — highest urgency first
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
