import base64
import re
from io import BytesIO
import ollama
import json
import time
from pdf2image import convert_from_path


def pdf_to_base64_images(pdf_path: str):
    images = convert_from_path(pdf_path, dpi=200)  # dpi=300 для лучшего качества
    base64_images = []

    for i, img in enumerate(images):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)
        print(f"Страница {i + 1}/{len(images)} обработана")

    return base64_images


def extract_json_from_text(text: str):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)

    if json_match:
        text = json_match.group(1)

    json_match = re.search(r'(\{.*\})', text, re.DOTALL)

    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        try:
            cleaned = re.sub(r'^.*?\{', '{', text, flags=re.DOTALL)
            return json.loads(cleaned.strip())
        except:
            print(f"Не удалось распарсить JSON. Сырой ответ: {text}")
            return None


def extract_with_llm(pdf_path: str):
    base64_images = pdf_to_base64_images(pdf_path)

    prompt1 = f"""
    Заполни форму и отдай её в виде JSON на основе документа из основных реквизитов сторон из данных, информации о самом документе (номера, даты и типа) и полной сводки о товарах и/или услугах, в том числе размер НДС.

    **Правила:**
    - Возвращай **ТОЛЬКО JSON**.
    - Тип документа нормирован, можно выбрать только один из списка - ["Универсальный передаточный документ", "Счёт-фактура", "Счёт", "Накладная", "Акт", "Прочее"].
    - Если значения нет в документе, то пиши null.
    - Не задавай уточняющие вопросы.
    - Не выдумывай данные.
    - Не пиши ничего лишнего.
    - При возможности используй режим think.
    
    Форма:
    ```json
    {{
        "document_type": Тип документа,
        "supplier_name": Полное наименование продавца/исполнителя,
        "inn": ИНН продавца/исполнителя,
        "kpp": КПП продавца/исполнителя,
        "number": Порядковый номер документа в формате числа,
        "date": Дата в формате ДД.MM.YYYY,
        "sum": Всего к оплате, сумма сделки,
        "vat": Сумма налога, предъявляемая покупателю,
        "currency": Буквенный код валюты по стандарту ISO 4217
    }}
    ```
    """

    stage1 = ollama.chat(
        model="qwen3.5:4b",
        messages=[{
            "role": "user",
            "content": prompt1,
            "images": base64_images
        }],
    )

    content1 = stage1["message"]["content"]

    return extract_json_from_text(content1)


if __name__ == "__main__":
    json_path = "../../data/benchmark/raw_docs/Счёт.pdf"

    start = time.perf_counter()
    result = extract_with_llm(json_path)
    end = time.perf_counter()
    print("\n=== ИТОГОВЫЙ РЕЗУЛЬТАТ ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Время выполнения: {end - start:.4f} секунд")

# qwen3.5:9b
# qwen3.5:4b
