import json
import requests
import functions


url = "http://localhost:8000/extract"
file_path = "data/benchmark/Отсканированный документ.pdf"

processed_path, _ = functions.preprocess_for_ocr(file_path)

with open(processed_path, "rb") as f:
    files = {"file": ("processed.pdf", f, "application/pdf")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Данные получены:", response.json())
    json_path = f"data/benchmark/predicts/paddle/{processed_path.split('/')[-1].removesuffix('.pdf')}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)
else:
    print(response.json())
