from pathlib import Path
import fitz


def extract_text_from_pdf(pdf_path, output_txt, output_html):
    doc = fitz.open(pdf_path)
    html_text = ""
    txt_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]

        text = page.get_text("text")
        txt_text += f"\nPage {page_num + 1}\n{text}\n"

        _html = page.get_text("html")
        html_text += f"\nPage {page_num+1}\n{_html}\n"

    Path(output_txt).write_text(txt_text.strip(), encoding="utf-8")
    Path(output_html).write_text(html_text.strip(), encoding="utf-8")
    print(f"Сохранено: {output_txt.name} ({len(txt_text)} символов)")
    print(f"Сохранено: {output_html.name} ({len(html_text)} символов)")

    return txt_text, html_text


if __name__ == "__main__":
    pdf_folder = Path("data/benchmark/raw_docs")
    output_text_folder = Path("data/benchmark/ground_truth/txt")
    output_html_folder = Path("data/benchmark/ground_truth/html")

    output_text_folder.mkdir(parents=True, exist_ok=True)
    output_html_folder.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_folder.glob("*.pdf"):
        print(f"Обработка: {pdf_file}")
        txt_output_path = output_text_folder / pdf_file.with_suffix(".txt").name
        html_output_path = output_html_folder / pdf_file.with_suffix(".html").name
        extract_text_from_pdf(pdf_file, txt_output_path, html_output_path)
