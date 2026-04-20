import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # Server
    DEBUG = os.getenv("DEBUG", "False") == "True"
    PORT = int(os.getenv("PORT", "5005"))
    HOST = os.getenv("HOST", "0.0.0.0")

    # Face recognition
    FACE_DISTANCE_THRESHOLD = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.8"))

    # Files
    DB_FILE = os.getenv("DB_FILE", "db.json")
