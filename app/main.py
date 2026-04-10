from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import io

from app.model import load_skin_model, predict_skin
from app.llm import generate_recommendation

app = FastAPI()

# 🧠 Load model once
model, class_names = load_skin_model()


@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):

    # 📥 read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 🤖 ML prediction
    disease, confidence = predict_skin(model, image, class_names)

    # 🧠 LLM recommendation
    llm_result = generate_recommendation(disease, confidence)

    return {
        "disease": disease,
        "confidence": float(confidence),
        **llm_result
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)