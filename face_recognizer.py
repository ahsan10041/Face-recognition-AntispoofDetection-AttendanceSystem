from deepface import DeepFace
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


class FaceRecognizer:
    """
    1:1 face verification using DeepFace with the FaceNet backbone.
    Embeddings (128-d) are stored per user and compared via cosine similarity.
    threshold=0.6 was chosen empirically: below this, matches are unreliable.
    """

    def __init__(self, db_path="data/users.pkl", threshold=0.6):
        self.db_path = db_path
        self.threshold = threshold
        self.database = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)

    def extract_embedding(self, face_image):
        """Run FaceNet and return a 128-d embedding, or None on failure."""
        try:
            result = DeepFace.represent(
                face_image,
                model_name="Facenet",
                enforce_detection=False,
            )
            return np.array(result[0]["embedding"])
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def register_user(self, name, face_image):
        if name in self.database:
            return False, "User already exists"
        emb = self.extract_embedding(face_image)
        if emb is None:
            return False, "Failed to extract face features"
        self.database[name] = {
            "embedding": emb,
            "registered_at": datetime.now().isoformat(),
        }
        self._save()
        return True, "User registered successfully"

    def recognize(self, face_image):
        """
        Compare face against all registered users.
        Returns (name, similarity) of the best match, or (None, similarity) if
        no match exceeds the threshold.
        """
        if not self.database:
            return None, 0.0

        emb = self.extract_embedding(face_image)
        if emb is None:
            return None, 0.0

        best_name = None
        best_sim = 0.0
        for name, data in self.database.items():
            sim = cosine_similarity(
                emb.reshape(1, -1),
                data["embedding"].reshape(1, -1),
            )[0][0]
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.threshold:
            return best_name, best_sim
        return None, best_sim

    def get_all_users(self):
        return list(self.database.keys())
