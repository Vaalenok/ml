import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output

# 1. Создаем словарь моделей (замена load_all_models)
# Это загрузит необходимые веса при первом вызове
model_dict = create_model_dict()

# 2. Инициализируем конвертер
# В новых версиях настройки передаются напрямую в конструктор
converter = PdfConverter(
    artifact_dict=model_dict
)

file_path = "../../data/benchmark/Отсканированный документ.pdf"
output_path = "../../data/benchmark/predictions"

# 3. Выполняем конвертацию
# Передаем список языков прямо в вызов
rendered = converter(file_path)

# 4. Сохраняем результат
base_filename = os.path.basename(file_path).rsplit(".", 1)[0]

# В новых версиях save_output принимает объект результата и путь
save_output(rendered, output_path, base_filename)

print(f"Готово! Файл {base_filename} обработан.")