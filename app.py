import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# =========================================================
# ⚠️ Preprocessing must match your training notebook exactly.
# Confirmed from deployment logs: VGG16 expects 256x256x3.
# =========================================================
CNN_INPUT_SIZE = (128, 128)
VGG16_INPUT_SIZE = (256, 256)

# =========================================================
# 📁 Add a `sample_images/` folder next to app.py with 4
# chest X-rays from your own Kaggle test set:
#   sample_images/sample_1.jpg   (e.g. a normal case)
#   sample_images/sample_2.jpg   (e.g. a normal case)
#   sample_images/sample_3.jpg   (e.g. a pneumonia case)
#   sample_images/sample_4.jpg   (e.g. a pneumonia case)
# Labels are intentionally generic so the model's answer
# isn't given away before analysis.
# =========================================================
SAMPLE_IMAGES = {
    "Sample 1": "sample_images/sample_1.jpg",
    "Sample 2": "sample_images/sample_2.jpg",
    "Sample 3": "sample_images/sample_3.jpg",
    "Sample 4": "sample_images/sample_4.jpg",
}

st.set_page_config(page_title="Pneumonia Triage AI", page_icon="🫁", layout="wide")

# ---------------------------------------------------------
# Light custom styling for a cleaner, more polished look
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall layout */
    .main .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1100px;}

    /* Headings */
    h1 {font-weight: 700; color: #0E7C7B; letter-spacing: -0.5px;}
    h2, h3 {font-weight: 600; color: #1A1A1A;}

    /* Sidebar */
    section[data-testid="stSidebar"] {background-color: #F0F5F5;}
    section[data-testid="stSidebar"] h1 {font-size: 1.4rem; color: #0E7C7B;}

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F7FAFA;
        border: 1px solid #E1E8E8;
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricValue"] {font-size: 1.7rem; color: #0E7C7B;}
    [data-testid="stMetricLabel"] {font-weight: 500;}

    /* Alerts */
    .stAlert {border-radius: 10px;}

    /* Progress bars */
    .stProgress > div > div > div > div {background-color: #0E7C7B;}

    /* Tables */
    table {border-radius: 8px; overflow: hidden;}

    /* Dividers */
    hr {margin: 1.5rem 0; border-color: #E1E8E8;}

    /* Radio buttons horizontal spacing */
    div[role="radiogroup"] {gap: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    path = hf_hub_download(repo_id="zainabfatima9/pnemonia-cnn-model", filename="cnn_model.h5")
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_vgg16_model():
    path = hf_hub_download(repo_id="zainabfatima9/pneumonia-vgg16-model", filename="vgg16_model.keras")
    return tf.keras.models.load_model(path)


def preprocess_for_cnn(pil_image):
    img = pil_image.convert("L").resize(CNN_INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)


def preprocess_for_vgg16(pil_image):
    img = pil_image.convert("RGB").resize(VGG16_INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


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
    colored = np.uint8(cm.jet(heatmap_arr / 255.0)[:, :, :3] * 255)
    base = np.array(pil_image.convert("RGB").resize(size))
    return Image.fromarray(np.uint8(base * 0.55 + colored * 0.45))


def run_full_analysis(image, cnn_model, vgg_model):
    cnn_input = preprocess_for_cnn(image)
    vgg_input = preprocess_for_vgg16(image)
    cnn_score = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
    vgg_score = float(vgg_model.predict(vgg_input, verbose=0)[0][0])
    last_conv = find_last_conv_layer(cnn_model)
    heatmap = make_gradcam_heatmap(cnn_input, cnn_model, last_conv) if last_conv else None
    overlay = overlay_heatmap(image, heatmap, size=(320, 320)) if heatmap is not None else None
    return cnn_score, vgg_score, overlay


def show_verdict(score, threshold=0.5):
    """Explicit if/else — avoids Streamlit 'magic' echoing a ternary expression."""
    if score >= threshold:
        st.error("🦠 Signs consistent with Pneumonia")
    else:
        st.success("✅ Normal")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 10px 0 4px 0;">
        <span style="font-size: 2.2rem;">🫁</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Pneumonia Triage AI")
st.sidebar.caption(
    "By **Zainab Fatima** · Medical Imaging Technology\n\n"
    "A prototype exploring whether AI can help prioritize which chest "
    "X-rays a radiologist reviews first — a triage aid, not a diagnostic tool."
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Diagnose an X-ray", "Model Comparison", "Triage Queue Simulator", "Future Vision"],
)

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "Home":
    st.title("🫁 AI-Assisted Pneumonia Triage")
    st.caption("From chest X-rays to a prioritized reading queue")

    st.markdown(
        """
        I'm a Medical Imaging Technology student — not a radiologist — and that's
        the actual starting point of this project. Technologists don't diagnose;
        we capture images and manage the workflow that decides how fast a scan
        gets seen. Often, that workflow is the real bottleneck: a technically
        perfect scan can still sit in a queue if there aren't enough specialists
        to read it in time.

        This project explores a narrower question than *"can AI diagnose
        pneumonia?"* — **can AI help make sure the most urgent X-rays are seen
        first?** I built and compared two models (a custom CNN and a VGG16
        transfer-learning model), added Grad-CAM interpretability, and deployed
        both as this live tool.
        """
    )

    st.subheader("Why this matters")
    c1, c2, c3 = st.columns(3)
    c1.metric("Global cases / year", "~450M")
    c2.metric("Global deaths / year", "~4M")
    c3.metric("Child deaths under 5 (2021)", "~500K+")
    st.caption(
        "Pakistan is consistently ranked by WHO/UNICEF and Global Burden of Disease "
        "estimates among the highest-burden countries for child pneumonia deaths "
        "worldwide, alongside India, Nigeria, and the DR Congo."
    )

    st.subheader("Why triage, not just detection")
    st.markdown(
        """
        Pneumonia can turn life-threatening within hours, especially in children
        and the elderly. Early diagnosis means earlier antibiotics, earlier oxygen
        support, and fewer complications. This tool doesn't replace a
        radiologist's diagnosis — it helps make sure the most urgent cases don't
        wait longest.
        """
    )

# ---------------------------------------------------------
# DIAGNOSE AN X-RAY
# ---------------------------------------------------------
elif page == "Diagnose an X-ray":
    st.title("Diagnose an X-ray")
    st.write("Upload an X-ray or pick a sample — both models' predictions and the Grad-CAM heatmap appear together.")

    source = st.radio("Image source", ["Upload my own", "Use a sample"], horizontal=True, label_visibility="collapsed")

    image = None
    if source == "Upload my own":
        uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        available = {name: path for name, path in SAMPLE_IMAGES.items() if os.path.exists(path)}
        if not available:
            st.warning("No sample images found. Add a `sample_images/` folder to enable this option.")
        else:
            choice = st.selectbox("Pick a sample", list(available.keys()))
            image = Image.open(available[choice])

    if image is not None:
        col_img, col_results = st.columns([1, 2])
        with col_img:
            st.image(image, caption="X-ray being analyzed", width="stretch")

        with st.spinner("Analyzing..."):
            cnn_model = load_cnn_model()
            vgg_model = load_vgg16_model()
            cnn_score, vgg_score, gradcam_overlay = run_full_analysis(image, cnn_model, vgg_model)

        with col_results:
            st.subheader("Predictions")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Custom CNN**")
                st.metric("Confidence", f"{cnn_score * 100:.1f}%")
                st.progress(min(max(cnn_score, 0.0), 1.0))
                show_verdict(cnn_score)
            with c2:
                st.markdown("**VGG16**")
                st.metric("Confidence", f"{vgg_score * 100:.1f}%")
                st.progress(min(max(vgg_score, 0.0), 1.0))
                show_verdict(vgg_score)

            if (cnn_score >= 0.5) == (vgg_score >= 0.5):
                st.info("✅ Both models agree.")
            else:
                st.warning("⚠️ Models disagree — a case that would benefit most from radiologist review.")

        st.divider()
        st.subheader("Grad-CAM: where the model is looking")
        if gradcam_overlay is not None:
            g1, g2 = st.columns(2)
            g1.image(image.resize((320, 320)), caption="Original")
            g2.image(gradcam_overlay, caption="Grad-CAM (Custom CNN)")
            st.caption(
                "Warmer regions show what most influenced the prediction — ideally "
                "the lung fields, not bone or background."
            )
        else:
            st.warning("Could not generate a Grad-CAM heatmap for this model.")

# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("A model trained from scratch vs. transfer learning on a limited medical imaging dataset.")

    st.table({
        "Model": ["Custom CNN", "VGG16 (Transfer Learning)"],
        "Test Accuracy": ["89%", "92%"],
        "Input": ["128×128 grayscale", "256×256 RGB"],
        "Approach": ["Trained from scratch", "Pretrained on ImageNet, fine-tuned"],
    })

    st.markdown(
        """
        **VGG16 is the stronger model and the recommended one for real use.**
        Its higher accuracy (92% vs. 89%) reflects a real advantage: it arrived
        already knowing general visual patterns from pretraining on millions of
        images, so it only needed to specialize for chest X-rays rather than
        learn to see from zero on a comparatively small medical dataset.

        In deployment, VGG16 would drive triage decisions, with the custom CNN
        kept as a lightweight baseline and for the Grad-CAM view — the same
        transfer-learning approach most production medical-imaging AI relies on.
        """
    )

# ---------------------------------------------------------
# TRIAGE QUEUE SIMULATOR
# ---------------------------------------------------------
elif page == "Triage Queue Simulator":
    st.title("Triage Queue Simulator")
    st.write("Instead of X-rays waiting in arrival order, an AI-prioritized queue reorders them by urgency.")

    use_demo = st.checkbox("Use demo queue (sample X-rays)", value=True)
    files_to_score = []

    if use_demo:
        for name, path in SAMPLE_IMAGES.items():
            if os.path.exists(path):
                files_to_score.append((name, Image.open(path)))
        if not files_to_score:
            st.warning("No sample images found. Add a `sample_images/` folder to enable the demo queue.")

    uploaded_files = st.file_uploader(
        "Or upload your own batch", type=["jpg", "jpeg", "png"], accept_multiple_files=True
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
            col1.image(img, width=90)
            with col2:
                st.write(f"**#{rank} — {name}**")
                st.progress(min(max(score, 0.0), 1.0))
            with col3:
                if score >= 0.5:
                    st.error(f"🦠 {score*100:.1f}% — priority review")
                else:
                    st.success(f"✅ {score*100:.1f}% — normal")

        st.caption(
            "The reordering — not a diagnosis — is the value: urgent scans reach "
            "a radiologist sooner, without changing who makes the final call."
        )

# ---------------------------------------------------------
# FUTURE VISION
# ---------------------------------------------------------
elif page == "Future Vision":
    st.title("Future Vision")
    st.write("A prototype, not a finished product. Realistic next steps toward clinical use:")

    st.markdown(
        """
        - **Workflow integration** — connect to a hospital's PACS/RIS so flagged
          scans are auto-prioritized in the radiologist's reading list.
        - **Multi-class detection** — bacterial vs. viral pneumonia, plus other
          findings (pleural effusion, TB-suggestive patterns).
        - **Severity scoring** — move from a binary flag to a graded urgency score.
        - **Multi-site validation** — test across hospitals, scanners, and
          populations beyond this single-source dataset.
        - **Edge deployment** — a lightweight version for clinics without an
          on-site radiologist, where triage delay matters most.
        - **Regulatory pathway** — FDA/CE clearance before any real-world use,
          even as a triage aid rather than a diagnostic tool.
        - **Combined dashboard** — pairing this with other imaging AI (e.g. a
          CT dose-optimization tool) into one department-wide triage view.
        """
    )

    st.info(
        "The model is the easy part. Making it fit how imaging departments "
        "actually work is the harder, more interesting problem — and the one "
        "worth building a career around."
    )
