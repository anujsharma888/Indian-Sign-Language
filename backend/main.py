from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

import numpy as np
import cv2
import io

from predictor import predict_frame

# ================= APP =================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LABEL MAP =================

label_map = {
    '0': 'I am ',
    '1': 'Vivek ',
    '2': 'Dholki'
}

# ================= ROUTES =================

@app.get("/")
def home():

    return {
        "message": "Sign Language API Running"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    image_np = np.array(image)

    frame = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    prediction = predict_frame(frame)

    if prediction is None:

        return {
            "success": False,
            "prediction": None
        }

    text = label_map.get(
        prediction["label"],
        prediction["label"]
    )

    return {
        "success": True,
        "prediction": text,
        "bbox": prediction["bbox"]
    }