import re
import os
import json
import requests
import projects.ocr.functions as functions

API_URL = "http://localhost:8081/v1/chat/completions"


def clean_json_string(text):
    text = re.sub(r'```json\s*|```', '', text)
    return text.strip()


def query_qwen(file_path, prompt):
    _, b64_images = functions.preprocess_for_ocr(file_path, dpi=150, need_pdf=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))

    base_filename = os.path.basename(file_path)
    for ext in ['.pdf', '.png', '.jpeg', '.jpg']:
        base_filename = base_filename.replace(ext, '')

    target_dir = os.path.join(project_root, "data", "models", "qwen")
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, f"qwen_{base_filename}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    for i, b64_img in enumerate(b64_images):
        print(f"Обработка страницы {i + 1} из {len(b64_images)}...")

        payload = {
            "model": "qwen3-vl-2b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}],
            "temperature": 0.01,
            "max_tokens": 2048
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=300)
            response.raise_for_status()

            raw_content = response.json()['choices'][0]['message']['content']
            clean_content = clean_json_string(raw_content)

            try:
                page_data = json.loads(clean_content)
            except json.JSONDecodeError:
                print(f"Ошибка: Невалидный JSON на странице {i + 1}. Сохраняю как текст.")
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

            print(f"Страница {i + 1} сохранена в файл.")

        except Exception as e:
            print(f"Ошибка на странице {i + 1}: {e}")

    print(f"\nОбработка завершена. Итоговый файл: {json_path}")


if __name__ == "__main__":
    task_prompt = """
        Проанализируй изображение и извлеки данные в формате JSON.
    
        1. Определи "document_type" (Паспорт, СНИЛС, Водительское удостоверение, Счет, УПД и др.).
        
        2. Если это ПЕРСОНАЛЬНЫЙ документ (Паспорт, СНИЛС и т.д.):
           - Извлеки: Фамилия, Имя, Отчество, серию и номер, дату рождения, дату выдачи и т.д.
        
        3. Если это ФИНАНСОВЫЙ документ (Счет, Накладная и т.д.):
           - Извлеки: ИНН, КПП, Поставщик, Номер документа, Дата, Сумма, НДС, Валюта, БИК, Расчетный счет.
           - Если есть таблица товаров/услуг, извлеки её в массив "table".
        
        4. Правила:
           - Используй только русский язык для ключей.
           - Если данных нет в документе, ставь null.
           - Верни ТОЛЬКО чистый JSON без Markdown и лишнего текста.
           
        Структура ответа:
        {
          "document_type": "...",
          "data": { ... }
        }
        
        Верни ТОЛЬКО JSON без Markdown.
        """

    query_qwen("../../data/docs/сканирование003 (2).pdf", task_prompt)