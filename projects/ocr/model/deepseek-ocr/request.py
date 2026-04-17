import json
from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model_name = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16
)
model = model.eval().cuda()

task_prompt = """
    <image>\n
    Extract all information from this document and return it in JSON format.

    Structure:

    {
      "document_type": "...",
      "data": {
        "Field Name": {"value": "value", "bbox": [x1, y1, x2, y2]},
        ...
      }
    }
    """

input_path = "../../data/benchmark/Отсканированный документ.pdf"
output_path = "../../data/models/deepseek"


images = convert_from_path(input_path, dpi=200)
final_results = {
    "document_name": Path(input_path).name,
    "pages_count": len(images),
    "pages_data": []
}

for i, img in enumerate(images):
    print(f"Обработка страницы {i + 1} из {len(images)}...")
    temp_image_path = Path(f"temp_page_{i + 1}.jpg").absolute()
    img.save(temp_image_path, "JPEG")

    try:
        res = model.infer(
            tokenizer=tokenizer,
            prompt=task_prompt,
            image_file=str(temp_image_path),
            output_path=str(output_path),
            base_size=1536,
            image_size=1536,
            crop_mode=False,
            save_results=False
        )

        try:
            page_json = json.loads(res) if isinstance(res, str) else res
        except:
            page_json = {"raw_output": res}

        final_results["pages_data"].append({
            "page_number": i + 1,
            "content": page_json
        })
    except Exception as e:
        print(f"Ошибка на странице {i + 1}: {e}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

output_file = output_path + f"_{Path(input_path).stem}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=4)

print(f"\nГотово! Результат собран в: {output_file}")
