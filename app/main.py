from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import io

from app.model import load_skin_model, predict_skin
from app.llm import generate_recommendation

app = FastAPI()

model, class_names = load_skin_model()

@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        disease, confidence = predict_skin(model, image, class_names)

        llm_result = generate_recommendation(disease, confidence)

        return {
            "disease": disease,
            "confidence": float(confidence),
            "recommendations": llm_result.get("recommendations", ""),
            "next_steps": llm_result.get("next_steps", ""),
            "tips": llm_result.get("tips", "")
        }

    except Exception as e:
        return {
            "error": "Something went wrong",
            "details": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )