import io
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CNN Image Classifier")
st.write("Upload an image and get a prediction.")

# ---------- CHOOSE ONE BACKEND: TensorFlow OR PyTorch ----------
USE_TF = True      # set to False if you use PyTorch
MODEL_LOCAL_PATH = "model/model.h5"   # TF .h5
# MODEL_LOCAL_PATH = "model/model.pt" # Torch .pt

# If your model is large and stored online (e.g., Google Drive or Hugging Face),
# download it once on startup. Example for Google Drive:
# import gdown
# FILE_ID = "1AbcXyz..."  # replace with your file id
# DEST = "model/model.h5"
# gdown.download(id=FILE_ID, output=DEST, quiet=False)

@st.cache_resource
def load_model():
    if USE_TF:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
            return ("tf", model)
        except Exception as e:
            st.warning(f"TF model not found/failed to load: {e}")
            return ("tf", None)
    else:
        try:
            import torch
            model = torch.load(MODEL_LOCAL_PATH, map_location="cpu")
            model.eval()
            return ("torch", model)
        except Exception as e:
            st.warning(f"Torch model not found/failed to load: {e}")
            return ("torch", None)

backend, model = load_model()

def preprocess_image(img: Image.Image, size=(224, 224), use_tf=True):
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    if use_tf:
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    else:
        # Torch: (1, C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image: Image.Image):
    # If you don't have a model yet, return a dummy result so the app still runs
    if model is None:
        return {"class": "No model loaded", "confidence": 0.0}

    if backend == "tf":
        import tensorflow as tf
        x = preprocess_image(image, size=(224, 224), use_tf=True)
        preds = model.predict(x)
        # Adjust this logic to your model's output shape / labels
        if preds.ndim == 2 and preds.shape[1] > 1:
            idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))
            label = f"class_{idx}"
        else:
            conf = float(preds[0][0])
            label = "positive" if conf >= 0.5 else "negative"
            conf = conf if label == "positive" else 1.0 - conf
        return {"class": label, "confidence": conf}

    else:
        import torch
        import torch.nn.functional as F
        x = preprocess_image(image, size=(224, 224), use_tf=False)
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            if logits.ndim == 2 and logits.shape[1] > 1:
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                conf = float(np.max(probs))
                label = f"class_{idx}"
            else:
                prob = torch.sigmoid(logits).cpu().numpy()[0][0].item()
                label = "positive" if prob >= 0.5 else "negative"
                conf = prob if label == "positive" else 1.0 - prob
        return {"class": label, "confidence": conf}

uploaded = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded image", use_column_width=True)
    with st.spinner("Running inference…"):
        result = predict(image)
    st.subheader("Prediction")
    st.write(f"**Class:** {result['class']}")
    st.write(f"**Confidence:** {result['confidence']:.2f}")
    st.caption("Tip: Replace the dummy label mapping with your dataset's class names.")

