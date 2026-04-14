import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import shutil
import uvicorn
from pydantic import BaseModel
from typing import List
import paddle


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


print("Compiled with CUDA:", paddle.is_compiled_with_cuda())
print("GPU count:", paddle.device.cuda.device_count())
print("Current device:", paddle.device.get_device())


os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ocr = PaddleOCR(lang="ru", use_gpu=True, use_angle_cls=True)
app = FastAPI(title="PaddleOCR API")


@app.post("/extract", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)):
    temp_dir = "/tmp/uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_{os.urandom(8).hex()}_{file.filename}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if not file.filename.endswith((".jpg", ".png", ".pdf")):
            raise HTTPException(
                status_code=400,
                detail="Неподдерживаемый формат файла. Используйте JPG, PNG или PDF."
            )

        results = ocr.ocr(temp_path)

        if not results:
            raise HTTPException(status_code=422, detail="Текст на изображении не обнаружен")

        output = []

        for i, page_data in enumerate(results):
            page_items = []

            if page_data is None:
                output.append({"page": i + 1, "items": []})
                continue

            for line in page_data:
                box = line[0]
                text = line[1][0]
                score = line[1][1]

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
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"Не удалось удалить временный файл {temp_path}: {cleanup_error}")


if __name__ == "__main__":
    print("Запуск API сервера...")
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=8000)
