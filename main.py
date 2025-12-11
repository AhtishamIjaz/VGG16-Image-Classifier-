# main.py
import os
import io
import json# Used for reading classes.json
from pathlib import Path# Used for reliable path resolution
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
# --- NEW IMPORT ---
from fastapi.middleware.cors import CORSMiddleware
# ------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- CORRECTED PATH DEFINITIONS for FLAT STRUCTURE ---
# Define the base directory as the directory containing this script (main.py)
BASE_DIR = Path(__file__).resolve().parent

# The files are DIRECTLY inside BASE_DIR
MODEL_PATH = BASE_DIR / "vgg16_final_model.h5"
CLASSES_PATH = BASE_DIR / "classes.json"
# ----------------------------------------------------

app = FastAPI(title="VGG16 Image Classifier API")

# --- ADDING CORS MIDDLEWARE (THE FIX) ---
origins = [
    "*",  # Allows all origins (for development). In production, list specific domains.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ----------------------------------------

## Load class names (Using JSON)
if not CLASSES_PATH.exists():
    # This checks the full, absolute path to classes.json
    raise RuntimeError(f"Class names file not found at {CLASSES_PATH}")
    
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    # 1. Load the JSON data into a map (e.g., {"almirah_dataset": 0, ...})
    CLASS_MAP = json.load(f)

# 2. CREATE THE REVERSE LOOKUP DICTIONARY (THE CRITICAL FIX)
# This maps the integer index (the model's output) to the class name string.
# Example: {0: "almirah_dataset", 1: "cardboard_box", ...}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()} 
# ----------------------------------------------------

# Load VGG16 model
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Function to convert uploaded file to model-ready array
def load_image_into_array(file_bytes):
    # Convert bytes to a numpy array
    img = image.load_img(io.BytesIO(file_bytes), target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if model expects 0-1
    return img_array

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_bytes = await file.read()
    try:
        img_array = load_image_into_array(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))
    
    # This line now works because CLASS_NAMES has been reversed:
    class_name = CLASS_NAMES[class_idx]
    
    confidence = float(preds[0][class_idx])

    return JSONResponse({
        "class_id": class_idx,
        "class_name": class_name,
        "confidence": confidence
    })