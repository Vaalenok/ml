import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import base64
import img2pdf
import re
import json
import requests
from pathlib import Path


def deskew(gray, target_dim=4000):
    try:
        h, w = gray.shape[:2]

        scale = target_dim / max(h, w)

        if scale != 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

            gray = cv2.resize(gray, (new_w, new_h), interpolation=interp)
            h, w = gray.shape[:2]

        margin_y, margin_x = int(h * 0.03), int(w * 0.03)
        roi = gray[margin_y:h - margin_y, margin_x:w - margin_x]
        binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 10)

        def score_angle(angle):
            M = cv2.getRotationMatrix2D((roi.shape[1] // 2, roi.shape[0] // 2), angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (roi.shape[1], roi.shape[0]),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

            return np.var(np.sum(rotated, axis=1))

        best_angle = max(np.arange(-45, 45.1, 2.0), key=score_angle)
        best_angle = max(np.arange(best_angle - 2, best_angle + 2.1, 0.5), key=score_angle)

        if abs(best_angle) < 0.3:
            return gray

        M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)

        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"Ошибка в deskew: {e}. Возвращаю оригинал.")
        return gray


def preprocess_for_ocr(file_path, output_dir="data/processed", do_deskew=True, dpi=300, need_pdf=True):
    try:
        if file_path.lower().endswith(".pdf"):
            pages_pil = convert_from_path(file_path, dpi=dpi, poppler_path=r'C:\Program Files\poppler\Library\bin')
            images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages_pil]
        else:
            img_data = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_COLOR)
            if img_data is None: raise ValueError("Не удалось декодировать изображение")
            images = [img_data]
    except Exception as e:
        print(f"Ошибка загрузки файла {file_path}: {e}")
        return None, []

    processed_bytes_list = []
    base64_list = []

    for i, img in enumerate(images):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if do_deskew:
                gray = deskew(gray)

            final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            success, buffer = cv2.imencode(".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success: continue

            img_bytes = buffer.tobytes()
            processed_bytes_list.append(img_bytes)
            base64_list.append(base64.b64encode(img_bytes).decode('utf-8'))
        except Exception as e:
            print(f"Ошибка на странице {i}: {e}")
            continue

    if not processed_bytes_list:
        print("Нет обработанных страниц для сохранения.")
        return None, []

    if need_pdf:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            base_name = os.path.basename(file_path)
            output_filename = f"proc_{os.path.splitext(base_name)[0]}.pdf"
            pdf_path = os.path.normpath(os.path.join(output_dir, output_filename)).replace("\\", "/")

            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(processed_bytes_list))

            print(f"Файл успешно сохранен: {pdf_path}")
            return pdf_path, base64_list
        except Exception as e:
            print(f"Ошибка сохранения PDF: {e}")
    return None, base64_list


def clean_json_string(text: str) -> str:
    text = re.sub(r'```json\s*|```', '', text).strip()

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]

    text = re.sub(r'(?s)("data":\s*\{.*?\})(?=.*\1)', '', text)
    text = re.sub(r'(?s)("table":\s*\[.*?\])(?=.*\1)', '', text)

    if text.count('{') == 1 and text.count('}') == 0:
        text += '}'
    elif text.count('{') > text.count('}'):
        text = text.rstrip(',') + '\n  }\n}'

    return text

def query_vlm_ocr(file_path, prompt, api_url, model_name, out_prefix="vlm", temp=0.0, tokens=3072, dpi=150, timeout=300):
    _, b64_images = preprocess_for_ocr(file_path, dpi=dpi, need_pdf=False)

    function_dir = Path(__file__).resolve().parent
    base_filename = Path(file_path).stem

    target_dir = function_dir / "data" / "models" / out_prefix
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"{out_prefix}_{base_filename}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    for i, b64_img in enumerate(b64_images):
        print(f"Обработка страницы {i + 1} из {len(b64_images)}...")

        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }],
            "temperature": temp,
            "max_tokens": tokens,
            "top_p": 0.9
        }

        try:
            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()

            raw_content = response.json()['choices'][0]['message']['content']

            if "lighton" in model_name.lower():
                structured_page = {
                    "page_number": i + 1,
                    "content": raw_content.strip()
                }
            else:
                clean_content = clean_json_string(raw_content)

                try:
                    page_data = json.loads(clean_content)
                except json.JSONDecodeError:
                    print(f"Невалидный JSON на странице {i + 1}")
                    page_data = {"error": "invalid_json", "raw": clean_content}

                structured_page = {
                    "page_number": i + 1,
                    "content": page_data
                }

            with open(json_path, "r+", encoding="utf-8") as f:
                current_data = json.load(f)
                current_data.append(structured_page)
                f.seek(0)
                json.dump(current_data, f, ensure_ascii=False, indent=2)
                f.truncate()

            print(f"Страница {i + 1} сохранена")

        except Exception as e:
            print(f"Ошибка на странице {i + 1}: {e}")
            structured_page = {
                "page_number": i + 1,
                "model": model_name,
                "error": str(e)
            }

            with open(json_path, "r+", encoding="utf-8") as f:
                current_data = json.load(f)
                current_data.append(structured_page)
                f.seek(0)
                json.dump(current_data, f, ensure_ascii=False, indent=2)
                f.truncate()

    print(f"\nОбработка завершена! Результат → {json_path}")
    return json_path