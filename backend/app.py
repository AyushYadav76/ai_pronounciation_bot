from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pronunciation_model import evaluate_pronunciation
import shutil
import os

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze/")
async def analyze_pronunciation(audio: UploadFile, expected_text: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, audio.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    result = evaluate_pronunciation(file_path, expected_text)
    os.remove(file_path)

    return JSONResponse(content=result)
