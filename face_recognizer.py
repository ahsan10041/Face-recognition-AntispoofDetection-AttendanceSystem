from deepface import DeepFace
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class FaceRecognizer:
    def __init__(self, db_path='data/users.pkl', threshold=0.6):
        self.db_path = db_path
        self.threshold = threshold
        self.database = self.load_database()
        
    def load_database(self):
        """Load user database"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_database(self):
        """Save user database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
    
    def extract_embedding(self, face_image):
        """Extract face embedding"""
        try:
            embedding = DeepFace.represent(
                face_image,
                model_name='Facenet',
                enforce_detection=False
            )
            return np.array(embedding[0]['embedding'])
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def register_user(self, name, face_image):
        """Register new user"""
        if name in self.database:
            return False, "User already exists"
        
        embedding = self.extract_embedding(face_image)
        
        if embedding is None:
            return False, "Failed to extract face features"
        
        self.database[name] = {
            'embedding': embedding,
            'registered_at': datetime.now().isoformat()
        }
        
        self.save_database()
        return True, "User registered successfully"
    
    def recognize(self, face_image):
        """
        Recognize face
        Returns: (name, confidence) or (None, 0)
        """
        if not self.database:
            return None, 0
        
        embedding = self.extract_embedding(face_image)
        
        if embedding is None:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for name, data in self.database.items():
            stored_emb = data['embedding']
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                stored_emb.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= self.threshold:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def get_all_users(self):
        """Get all registered users"""
        return list(self.database.keys())
    
    def get_user_count(self):
        """Get total user count"""
        return len(self.database)