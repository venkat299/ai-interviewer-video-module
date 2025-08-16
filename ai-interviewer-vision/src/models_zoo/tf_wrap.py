"""TensorFlow model wrapper stub."""
from __future__ import annotations

try:  # pragma: no cover - optional
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def load_tf_model(path: str):
    if tf is None:
        return None
    return tf.saved_model.load(path)
