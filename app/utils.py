import numpy as np
from deepface import DeepFace

DIST_THRESHOLD = 0.4

def embed_face(img_bgr):
    """Convert a face image into 128D embedding vector."""
    try:
        embedding = DeepFace.represent(
            img_path=img_bgr, model_name="Facenet", enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding)
    except Exception:
        return None

def cosine_dist(a, b):
    """Cosine distance between two embeddings."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)
