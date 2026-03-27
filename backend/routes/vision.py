import os
import base64
from io import BytesIO

import numpy as np
import requests as req_lib
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "cnn_model.h5")

_cnn_model = None


def get_model():
    global _cnn_model
    if _cnn_model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        from tensorflow.keras.models import load_model
        _cnn_model = load_model(MODEL_PATH)
        print(f"CNN loaded. Output: {_cnn_model.output_shape}")
    return _cnn_model


def compute_gradcam(model, img_array):
    """GradCAM for Sequential model — returns base64 RGBA PNG heatmap."""
    try:
        import tensorflow as tf

        # Find index of last Conv2D layer
        last_idx = None
        for i, layer in enumerate(model.layers):
            if "conv2d" in layer.name:
                last_idx = i

        if last_idx is None:
            return None

        # Build a functional sub-model: input → last conv output
        inp = tf.keras.Input(shape=(128, 128, 3))
        x = inp
        for layer in model.layers[: last_idx + 1]:
            x = layer(x)
        conv_model = tf.keras.Model(inputs=inp, outputs=x)

        img_t = tf.cast(img_array, tf.float32)

        # Watch conv output, compute gradient of loss w.r.t. it
        with tf.GradientTape() as tape:
            conv_out = conv_model(img_t)
            tape.watch(conv_out)
            # Pass through remaining layers
            x2 = conv_out
            for layer in model.layers[last_idx + 1 :]:
                x2 = layer(x2)
            loss = tf.reduce_mean(x2)

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            print("GradCAM: gradient is None")
            return None

        # Pool gradients over spatial dims → (channels,)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_arr = conv_out[0].numpy()          # (H, W, C)
        heatmap = np.dot(conv_arr, pooled)      # (H, W)

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize to 128×128
        hm_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (128, 128), Image.BILINEAR
        )
        hm = np.array(hm_img).astype(np.float32) / 255.0

        # Jet colormap (blue→cyan→green→yellow→red)
        r = np.clip(1.5 - np.abs(hm * 4.0 - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(hm * 4.0 - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(hm * 4.0 - 1.0), 0, 1)
        alpha = np.clip(hm * 210 + 40, 0, 240).astype(np.uint8)

        rgba = np.stack(
            [
                (r * 255).astype(np.uint8),
                (g * 255).astype(np.uint8),
                (b * 255).astype(np.uint8),
                alpha,
            ],
            axis=-1,
        )

        buf = BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        result = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"GradCAM OK, b64 length: {len(result)}")
        return result

    except Exception as e:
        print(f"GradCAM failed: {e}")
        import traceback; traceback.print_exc()
        return None


def load_image(image_url=None, image_base64=None):
    if image_base64:
        data = base64.b64decode(image_base64)
        return Image.open(BytesIO(data)).convert("RGB")
    if image_url.startswith("http"):
        r = req_lib.get(image_url, timeout=15)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    path = image_url if os.path.isabs(image_url) else os.path.join(BASE_DIR, image_url)
    return Image.open(path).convert("RGB")


class VisionRequest(BaseModel):
    image_url: str = None
    image_base64: str = None


@router.post("/vision")
def analyze_satellite_feed(req: VisionRequest):
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    try:
        image = load_image(req.image_url, req.image_base64)
        image_resized = image.resize((128, 128))
        x = np.expand_dims(np.array(image_resized, dtype=np.float32) / 255.0, axis=0)

        pred = model.predict(x, verbose=0)
        pred_sq = np.squeeze(pred)

        gradcam_b64 = compute_gradcam(model, x)

        if pred_sq.ndim == 2:
            flood_mask = (pred_sq > 0.5).astype(np.uint8)
            flood_pct = float(np.mean(flood_mask) * 100)
            return {
                "status": "OK",
                "mode": "segmentation",
                "flood_percentage": round(flood_pct, 2),
                "flood_state": "FLOODED" if flood_pct > 10 else "SAFE",
                "gradcam": gradcam_b64,
            }

        flood_prob = float(pred_sq if pred_sq.ndim == 0 else pred_sq[0])
        return {
            "status": "OK",
            "mode": "classification",
            "flood_percentage": round(flood_prob * 100, 2),
            "flood_state": "FLOODED" if flood_prob > 0.5 else "SAFE",
            "gradcam": gradcam_b64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
