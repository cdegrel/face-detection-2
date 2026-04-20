import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from PIL import Image
from src.config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(device=device)

def detect_faces(frame):
    """Detect faces from frame using MTCNN."""
    try:
        boxes, probs = mtcnn.detect(frame, landmarks=False)
        return boxes, probs, frame
    except Exception as e:
        print(f"Detection error: {e}")
        return None, None, frame

def extract_embeddings(frame, boxes):
    """Extract FaceNet embeddings from detected faces."""
    if boxes is None or len(boxes) == 0:
        return []

    embeddings = []
    rgb_frame = frame

    for box in boxes:
        try:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_crop = rgb_frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_crop = np.array(Image.fromarray(face_crop).resize((160, 160), Image.LANCZOS))
            face_tensor = transforms.ToTensor()(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy()
            embeddings.append(embedding[0])
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            continue

    return embeddings
