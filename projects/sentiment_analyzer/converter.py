import csv
import os
import pandas as pd
import re
from io import StringIO

INPUT_FILES = [
    "data/html/Москва_Мое_сокровище_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov.html",
    "data/html/Московская_область_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov.html",
    "data/html/Подарок_малышу_Челябинск_CSV_файлов_в_виде_таблицы_онлайн___vk.html",
    "data/html/Татарстан_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov_net.html",
    "data/html/Ульяновская_область_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov.html",
    "data/html/Шкатулка_Расту_в_Югре_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov.html",
    "data/html/Ямал_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov_net.html",
    "data/html/Санкт_Петербург_CSV_файлов_в_виде_таблицы_онлайн___vk_barkov_net.html"
]
OUTPUT_FILES = [
    "data/csv/Москва_Мое_сокровище.csv",
    "data/csv/Московская_область.csv",
    "data/csv/Подарок_малышу_Челябинск.csv",
    "data/csv/Татарстан.csv",
    "data/csv/Ульяновская_область.csv",
    "data/csv/Шкатулка_Расту_в_Югре.csv",
    "data/csv/Ямал.csv",
    "data/csv/Санкт_Петербург.csv"
]

def convert_html_table_to_csv(html_content: str) -> str:
    try:
        dfs = pd.read_html(StringIO(html_content), encoding="utf-8", header=0)

        if not dfs:
            raise ValueError("В HTML-файле не найдено ни одной таблицы.")

        df = dfs[0]

        if len(df.columns) > 1:
            df = df.iloc[:, 1:].copy()

        new_columns = []

        for col in df.columns:
            clean_col = str(col).replace("скопировать", "").strip()
            new_columns.append(clean_col)

        df.columns = new_columns

        def clean_cell(cell):
            if pd.isna(cell):
                return cell

            cell_str = str(cell)
            cell_str = re.sub(r"<.*?>", "", cell_str)
            cell_str = (cell_str.replace("\n", " ").replace("\r", " ")
                        .replace("<br>", " ").replace("<br/>", " "))
            cell_str = re.sub(r"\s+", " ", cell_str).strip()

            return cell_str

        for col in df.columns:
            df[col] = df[col].apply(clean_cell)

        csv_data = df.to_csv(index=False, sep=",", encoding="utf-8", quoting=csv.QUOTE_ALL)

        return csv_data

    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")
        return ""

if __name__ == "__main__":
    for file, output_file in zip(INPUT_FILES, OUTPUT_FILES):
        if not os.path.exists(file):
            print(f"Входной файл '{file}' не найден.")
        else:
            with open(file, "r", encoding="utf-8") as f:
                _html_content = f.read()

            csv_output = convert_html_table_to_csv(_html_content)

            if csv_output:
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    f.write(csv_output)

                print(f"Данные сохранены в файл: {output_file}")
            else:
                print("Конвертация не удалась.")
