import json
import os
import time
import numpy as np
from src.config import Config

def load_faces_db():
    """Load face database from JSON file."""
    if os.path.exists(Config.DB_FILE):
        try:
            with open(Config.DB_FILE, 'r') as f:
                db = json.load(f)
                return db.get('faces', {})
        except Exception as e:
            print(f"Error loading faces DB: {e}")
            return {}
    return {}

def save_faces_db(faces):
    """Save face database to JSON file."""
    try:
        if os.path.exists(Config.DB_FILE):
            with open(Config.DB_FILE, 'r') as f:
                db = json.load(f)
        else:
            db = {'faces': {}, 'rtsp_streams': {}}

        db['faces'] = faces

        with open(Config.DB_FILE, 'w') as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"Error saving faces DB: {e}")

def recognize_face(embedding, db):
    """Recognize face by comparing embeddings with database.

    Returns (name, distance) or (None, None) if no match found.
    """
    if embedding is None or len(db) == 0:
        return None, None

    best_match = None
    best_distance = Config.FACE_DISTANCE_THRESHOLD

    for name, face_data in db.items():
        if isinstance(face_data, dict) and 'embedding' in face_data:
            try:
                stored_embedding = np.array(face_data['embedding'])
                distance = np.linalg.norm(embedding - stored_embedding)

                if distance < best_distance:
                    best_distance = distance
                    best_match = name
            except Exception as e:
                print(f"Error comparing embeddings for {name}: {e}")
                continue

    return best_match, best_distance

def save_face_embedding(name, embedding):
    """Save a face embedding to the database."""
    db = load_faces_db()

    if name in db:
        return False, f'Face "{name}" already exists'

    db[name] = {
        'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        'timestamp': time.time()
    }
    save_faces_db(db)
    return True, f'Face "{name}" saved successfully'

def delete_face(name):
    """Delete a face from the database."""
    db = load_faces_db()

    if name not in db:
        return False, f'Face "{name}" not found'

    del db[name]
    save_faces_db(db)
    return True, f'Face "{name}" deleted'

def get_all_faces():
    """Get all face names from database."""
    db = load_faces_db()
    return list(db.keys())
