import requests
import functions


url = "http://localhost:8000/extract"
file_path = "data/docs/сканирование (2).pdf"

processed_path = functions.preprocess_for_ocr(file_path)

with open(processed_path, "rb") as f:
    files = {"file": ("processed.pdf", f, "application/pdf")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Данные получены:", response.json())
else:
    print(response.json())
