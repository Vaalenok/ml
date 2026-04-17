from pathlib import Path
import json
import jiwer
import pandas as pd
from typing import Dict
import re


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_html_text(html_text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', html_text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_json(json_path: str) -> Dict[int, str]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    pages = {}

    if isinstance(data, list) and "page_number" in data[0]:
        for item in data:
            page_num = item.get("page_number")
            content = item.get("content", "")

            if isinstance(content, dict):
                content = str(content)

            pages[page_num] = clean_text(content)
    elif isinstance(data, list) and "content" in data[0]:
        for item in data:
            page_num = item.get("page_number")
            content = item.get("content", "")

            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)

            pages[page_num] = clean_text(str(content))
    elif isinstance(data, dict) and "data" in data:
        for page in data.get("data", []):
            page_num = page.get("page")
            text = ""

            for item in page.get("items", []):
                text += item.get("text", "") + " "

            pages[page_num] = clean_text(text)

    return pages


def calculate_metrics(gt: str, pred: str) -> Dict:
    cer = jiwer.cer(gt, pred)
    wer = jiwer.wer(gt, pred)

    return {
        "CER": round(cer, 5),
        "WER": round(wer, 5),
        "Char_Acc": round((1 - cer) * 100, 3),
        "Word_Acc": round((1 - wer) * 100, 3),
    }


if __name__ == "__main__":
    gt_txt_folder = Path("data/benchmark/ground_truth/txt")
    gt_html_folder = Path("data/benchmark/ground_truth/html")
    pred_folder = Path("data/benchmark/predictions")

    results = []

    for pred in pred_folder.glob("*.json"):
        model_name = pred.stem.split('_')[0]
        pred_pages = extract_text_from_json(str(pred))

        for gt_text_file in gt_txt_folder.glob("*.txt"):
            doc_name = gt_text_file.stem

            gt_text = clean_text(gt_text_file.read_text(encoding="utf-8"))
            metrics = calculate_metrics(gt_text, pred_pages[int(doc_name)])
            results.append({
                "Page": int(doc_name),
                "GroundTruth": "TEXT",
                "Model": model_name,
                **metrics
            })

        all_ground_truth = "\n\n".join(
            clean_text(file.read_text(encoding="utf-8"))
            for file in gt_txt_folder.glob("*.txt")
        )
        full_pred = " ".join(pred_pages.values())
        full_metrics = calculate_metrics(all_ground_truth, full_pred)

        results.append({
            "Page": "TOTAL",
            "GroundTruth": "TEXT",
            "Model": model_name,
            **full_metrics
        })

        for gt_html_file in gt_html_folder.glob("*.html"):
            doc_name = gt_html_file.stem

            gt_html = clean_html_text(gt_html_file.read_text(encoding="utf-8"))
            metrics = calculate_metrics(gt_html, pred_pages[int(doc_name)])
            results.append({
                "Page": int(doc_name),
                "GroundTruth": "HTML",
                "Model": model_name,
                **metrics
            })

        all_ground_truth = "\n\n".join(
            clean_html_text(file.read_text(encoding="utf-8"))
            for file in gt_html_folder.glob("*.html")
        )
        full_pred = " ".join(pred_pages.values())
        full_metrics = calculate_metrics(all_ground_truth, full_pred)

        results.append({
            "Page": "TOTAL",
            "GroundTruth": "HTML",
            "Model": model_name,
            **full_metrics
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Page", "CER"])

    print("\n" + df.to_string(index=False))

    # df.to_excel("benchmark_results.xlsx", index=False)
    # df.to_csv("benchmark_results.csv", index=False)

    summary = df[df["Page"] == "TOTAL"].groupby("Model")[["CER", "WER", "Char_Acc", "Word_Acc"]].mean().round(4)
    print("\nСредние метрики по моделям (по всему документу):")
    print(summary)
