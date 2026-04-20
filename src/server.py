import json
import os
from io import BytesIO

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av.video.frame import VideoFrame
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.config import Config
from src.core.detection import detect_faces, extract_embeddings
from src.core.recognition import (
    delete_face,
    get_all_faces,
    load_faces_db,
    recognize_face,
    save_face_embedding,
)

app = FastAPI(title="Face Recognition System", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Interface not found</h1>"


@app.get("/faces")
async def get_faces():
    """Get all recognized faces."""
    return {"faces": get_all_faces()}


@app.post("/faces")
async def add_face(name: str, file: UploadFile = File(...)):
    """Add a new face to database."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Name required")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        frame = np.array(img)

        boxes, _, _ = detect_faces(frame)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        embeddings = extract_embeddings(frame, boxes)
        if len(embeddings) == 0:
            raise HTTPException(status_code=400, detail="Could not extract embedding")

        success, message = save_face_embedding(name, embeddings[0])
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/faces/{name}")
async def remove_face(name: str):
    """Delete a face from database."""
    success, message = delete_face(name)
    if success:
        return {"status": "success", "message": message}
    return {"status": "error", "message": message}


def load_rtsp_streams():
    """Load RTSP streams from database."""
    try:
        with open(Config.DB_FILE, "r") as f:
            db = json.load(f)
            return db.get("rtsp_streams", {})
    except Exception as e:
        print(f"Error loading RTSP streams: {e}")
        return {}


def save_rtsp_streams(streams):
    """Save RTSP streams to database."""
    try:
        # Load existing database
        if os.path.exists(Config.DB_FILE):
            with open(Config.DB_FILE, "r") as f:
                db = json.load(f)
        else:
            db = {"faces": {}, "rtsp_streams": {}}

        # Update streams
        db["rtsp_streams"] = streams

        # Save
        with open(Config.DB_FILE, "w") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"Error saving RTSP streams: {e}")


@app.get("/rtsp_streams")
async def get_rtsp_streams():
    """Get all RTSP streams."""
    streams = load_rtsp_streams()
    return {"streams": streams}


@app.post("/rtsp_streams")
async def add_rtsp_stream(name: str = None, url: str = None):
    """Add a new RTSP stream."""
    if not name or not url:
        raise HTTPException(status_code=400, detail="Name and URL required")

    name = name.strip()
    url = url.strip()

    streams = load_rtsp_streams()
    if name in streams:
        raise HTTPException(status_code=400, detail=f'Stream "{name}" already exists')

    streams[name] = url
    save_rtsp_streams(streams)
    return {"status": "success", "message": f'Stream "{name}" added'}


@app.delete("/rtsp_streams/{name}")
async def delete_rtsp_stream(name: str):
    """Delete an RTSP stream."""
    streams = load_rtsp_streams()
    if name not in streams:
        raise HTTPException(status_code=400, detail=f'Stream "{name}" not found')

    del streams[name]
    save_rtsp_streams(streams)
    return {"status": "success", "message": f'Stream "{name}" deleted'}


class RTSPStreamTrack(VideoStreamTrack):
    """Track that reads from an RTSP stream."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.container = av.open(url, options={"rtsp_transport": "tcp"})
        self.stream = self.container.streams.video[0]

    async def recv(self):
        frame = None
        for packet in self.container.demux(self.stream):
            for f in packet.decode():
                frame = f
                break
            if frame:
                break

        if frame is None:
            return frame

        img = frame.to_ndarray(format="bgr24")
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self.pts
        frame.time_base = self.time_base
        return frame


rtc_connections = {}


@app.post("/rtsp_webrtc")
async def rtsp_webrtc(data: dict):
    """WebRTC endpoint for RTSP streams."""
    try:
        stream_url = data.get("url")
        offer_sdp = data.get("sdp")

        if not stream_url or not offer_sdp:
            raise HTTPException(status_code=400, detail="URL and SDP required")

        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

        pc = RTCPeerConnection()
        rtc_connections[id(pc)] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await pc.close()
                rtc_connections.pop(id(pc), None)

        try:
            track = RTSPStreamTrack(stream_url)
            pc.addTrack(track)
        except Exception as e:
            print(f"Error creating RTSP track: {e}")
            raise HTTPException(
                status_code=400, detail=f"Failed to connect to RTSP stream: {e}"
            )

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"WebRTC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_embedding(img_pil):
    frame = np.array(img_pil)
    boxes, _, _ = detect_faces(frame)
    if boxes is None or len(boxes) == 0:
        return None
    embeddings = extract_embeddings(frame, boxes)
    return embeddings[0] if embeddings else None


@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compare two faces and return similarity score."""
    try:
        img1_pil = Image.open(BytesIO(await file1.read())).convert("RGB")
        img2_pil = Image.open(BytesIO(await file2.read())).convert("RGB")

        emb1 = _extract_embedding(img1_pil)
        emb2 = _extract_embedding(img2_pil)

        if emb1 is None or emb2 is None:
            raise HTTPException(
                status_code=400, detail="Face not detected in one or both images."
            )

        distance = np.linalg.norm(emb1 - emb2)
        match = distance < Config.FACE_DISTANCE_THRESHOLD

        return {"match": bool(match), "distance": round(float(distance), 4)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Recognize a face against the database."""
    try:
        img_pil = Image.open(BytesIO(await file.read())).convert("RGB")
        embedding = _extract_embedding(img_pil)

        if embedding is None:
            raise HTTPException(status_code=400, detail="Face not detected in image.")

        db = load_faces_db()
        name, distance = recognize_face(embedding, db)

        return {
            "name": name if name else "Unknown",
            "distance": round(float(distance), 4) if distance else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint to stream face detections from client frames."""
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                if not data:
                    continue

                try:
                    img = Image.open(BytesIO(data))
                    frame = np.array(img)
                except Exception as e:
                    print(f"Image decode error: {e}")
                    await websocket.send_json({"faces": []})
                    continue

                try:
                    boxes, _, _ = detect_faces(frame)
                except Exception as e:
                    print(f"Detection error: {e}")
                    await websocket.send_json({"faces": []})
                    continue

                faces_with_names = []
                if boxes is not None and len(boxes) > 0:
                    try:
                        embeddings = extract_embeddings(frame, boxes)
                        db = load_faces_db()
                        h, w = frame.shape[:2]

                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            name = "Unknown"
                            if i < len(embeddings):
                                try:
                                    recognized_name, distance = recognize_face(
                                        embeddings[i], db
                                    )
                                    if recognized_name:
                                        name = recognized_name
                                except Exception as e:
                                    print(f"Recognition error: {e}")

                            faces_with_names.append(
                                {
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2),
                                    "name": name,
                                }
                            )
                    except Exception as e:
                        print(f"Processing error: {e}")

                try:
                    await websocket.send_json({"faces": faces_with_names})
                except Exception as e:
                    print(f"Send error: {e}")
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Frame error: {e}")
                continue
    except Exception as e:
        print(f"WebSocket detect fatal error: {e}")
