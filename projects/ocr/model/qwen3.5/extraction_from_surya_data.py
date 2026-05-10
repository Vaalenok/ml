import re
import ollama
import json
import time


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


def extract_with_llm(ocr_data: dict):
    items = []
    for item in ocr_data.get("text_lines", []):
        text = item.get("text", "").strip()

        if text:
            x_center = (item["bbox"][0] + item["bbox"][2]) / 2
            y_center = (item["bbox"][1] + item["bbox"][3]) / 2
            items.append({
                "text": text,
                "x": round(x_center, 0),
                "y": round(y_center, 0),
            })

    items.sort(key=lambda _item: (_item["y"], _item["x"]))

    full_text = "\n".join(f"[x={item['x']}, y={item['y']}] {item['text']}" for item in items)

    prompt1 = f"""
    Составь отчёт из основных реквизитов сторон из данных, информации о самом документе (номера, даты и типа) и полной сводки о товарах и/или услугах, описанных в данных, в том числе размер НДС.
    
    **Правила:**
    - Данные разделены на строки. Каждая строка имеет следующий вид - [x=<положение по x, число>, y=<положение по y, число>] <текст>.
    - В контексте данного задания положение блока текста по x и y имеет основополагающее значение - это позиция блока на документе.
    - Учитывай не только смысл текста, но и его позицию относительно документа и других блоков.
    - Дата и номер документа **всегда указаны в данных**.
    - Тип документа **всегда явно указан в данных**.
    - Тип документа нормирован, можно выбрать только один из списка - ["Универсальный передаточный документ", "Счёт-фактура", "Счёт", "Накладная", "Акт", "Прочее"].
    - Не задавай уточняющие вопросы.
    - Не выдумывай данные.
    - Не пиши ничего лишнего.
    - При возможности используй режим think.
    
    Данные:
    {full_text}
    """

    stage1 = ollama.chat(
        model="qwen3.5:4b",
        messages=[{
            "role": "user",
            "content": prompt1
        }],
    )

    content1 = stage1["message"]["content"]
    print(content1)

    prompt2 = f"""
    На основе отчёта по документу заполни форму и отдай её в виде json.
    
    **Правила:**
    - Если значения нет в отчёте, то пиши null.
    
    Форма:
    ```json
    {{
        "document_type": Тип документа - "Универсальный передаточный документ" | "Счёт-фактура" | "Счёт" | "Накладная" | "Акт" | "Прочее",
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
    
    Отчёт:
    {content1}
    """

    stage2 = ollama.chat(
        model="qwen3.5:4b",
        messages=[{
            "role": "user",
            "content": prompt2
        }]
    )

    content2 = stage2["message"]["content"]

    return extract_json_from_text(content2)


if __name__ == "__main__":
    json_path = "../../data/benchmark/predictions/surya_proc_Отсканированный документ.json"

    with open(json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    start = time.perf_counter()
    result = extract_with_llm(ocr_data)
    end = time.perf_counter()
    print("\n=== ИТОГОВЫЙ РЕЗУЛЬТАТ ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Время выполнения: {end - start:.4f} секунд")

# qwen3.5:9b
# qwen3.5:4b
