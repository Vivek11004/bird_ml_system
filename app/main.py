from fastapi import FastAPI, UploadFile, File
import shutil
import os
from app.video_analyzer import analyze_video

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze_video_api(
    video: UploadFile = File(...),
    fps_sample: int = 5
):
    os.makedirs("outputs", exist_ok=True)

    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    counts, weights = analyze_video(temp_path, fps_sample)

    os.remove(temp_path)

    return {
        "counts": counts,
        "weight_estimates": weights,
        "unit": "relative_weight_index",
        "artifacts": ["outputs/annotated_video.mp4"]
    }
