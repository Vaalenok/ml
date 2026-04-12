import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import shutil
import uvicorn
from pydantic import BaseModel, validator
from typing import List


class OCRItem(BaseModel):
    box: List[List[float]]
    text: str
    confidence: float

class OCRPage(BaseModel):
    page: int
    items: List[OCRItem]

class OCRResponse(BaseModel):
    status: str
    data: List[OCRPage]


os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

app = FastAPI(title="PaddleOCR API")
ocr = PaddleOCR(lang="ru")


@app.post("/extract")
async def extract_text(file: UploadFile = File(...), response_model=OCRResponse):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if not file.filename.endswith((".jpg", ".png", ".pdf")):
            raise HTTPException(
                status_code=400,
                detail="Неподдерживаемый формат файла. Используйте JPG, PNG или PDF."
            )

        results = ocr.predict(temp_path)

        if not results:
            raise HTTPException(status_code=422, detail="Текст на изображении не обнаружен")

        output = []

        for i, page_data in enumerate(results):
            page_items = []

            if isinstance(page_data, dict):
                texts = page_data.get("rec_texts", [])
                scores = page_data.get("rec_scores", [])
                boxes = page_data.get("dt_polys", [])

                for text, score, box in zip(texts, scores, boxes):
                    if isinstance(box, np.ndarray):
                        box = box.tolist()

                    page_items.append({
                        "box": box,
                        "text": text,
                        "confidence": float(score)
                    })

            output.append({
                "page": i + 1,
                "items": page_items
            })

        return OCRResponse(status="success", data=output)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("Запуск API сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
