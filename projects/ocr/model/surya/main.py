import torch
from PIL import ImageDraw, Image
from pdf2image import convert_from_path
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import
from pathlib import Path
import json


Image.MAX_IMAGE_PIXELS = None
torch.cuda.empty_cache()


pdf_path = "../../data/processed/proc_Отсканированный документ.pdf"
output_dir = "../../data/bbox"
json_dir = "../../data/benchmark/predictions"

Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(json_dir).mkdir(parents=True, exist_ok=True)


pages = convert_from_path(
    pdf_path,
    dpi=100,
    poppler_path=r'C:\Program Files\poppler\Library\bin'
)

print(f"Найдено {len(pages)} страниц")


foundation = FoundationPredictor()


recognition_predictor = RecognitionPredictor(foundation)
detection_predictor = DetectionPredictor()

rec_predictions = recognition_predictor(pages, det_predictor=detection_predictor)

del recognition_predictor
del detection_predictor
torch.cuda.empty_cache()


layout_predictor = LayoutPredictor(foundation)
layout_predictions = layout_predictor(pages)

del layout_predictor


all_pages_data = []

for page_idx, (image, rec_result, layout_result) in enumerate(
    zip(pages, rec_predictions, layout_predictions)
):
    draw = ImageDraw.Draw(image)

    print(f"\nСтраница {page_idx + 1}")

    page_data = {
        "page": page_idx + 1,
        "items": [],
        "layout_blocks": []
    }

    for block in layout_result.bboxes:
        x1, y1, x2, y2 = block.bbox
        block_type = block.label.upper()

        color = (0, 0, 255) if block_type == "TABLE" else \
                (255, 0, 255) if block_type == "FIGURE" else \
                (0, 255, 255) if block_type == "LIST" else \
                (255, 140, 0) if "TITLE" in block_type or "HEADER" in block_type else \
                (100, 100, 100)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=12)
        draw.text((x1, max(0, y1 - 50)), f"[{block_type}]", fill=color)

        page_data["layout_blocks"].append({
            "type": block_type,
            "bbox": [round(x, 2) for x in [x1, y1, x2, y2]],
            "confidence": round(float(block.confidence), 4) if hasattr(block, 'confidence') else None
        })

    for line in rec_result.text_lines:
        x1, y1, x2, y2 = line.bbox
        conf = line.confidence
        color = (0, 255, 0) if conf > 0.75 else (255, 165, 0) if conf > 0.5 else (255, 0, 0)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 25)), f"{line.text} ({conf:.2f})", fill=color)

        page_data["items"].append({
            "text": line.text,
            "confidence": round(float(conf), 4),
            "bbox": [round(float(x), 2) for x in [x1, y1, x2, y2]]
        })

    output_img = Path(output_dir) / f"page_{page_idx + 1:02d}_with_blocks.jpg"
    image.save(output_img, quality=95)

    all_pages_data.append(page_data)

json_path = Path(json_dir) / f"surya_{Path(pdf_path).stem}.json"

final_json = {
    "document": Path(pdf_path).name,
    "total_pages": len(pages),
    "data": all_pages_data
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(final_json, f, ensure_ascii=False, indent=2)

print(f"Изображения сохранены в: {output_dir}")
print(f"JSON сохранён: {json_path}")
