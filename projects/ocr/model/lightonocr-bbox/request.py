import projects.ocr.functions as functions

task_prompt = """
    Извлеки весь текст из документа максимально точно и полно.
    Не добавляй никаких объяснений, JSON, markdown или таблиц.
    Просто верни весь распознанный текст.
    """

functions.query_vlm_ocr(
    file_path="../../data/img/prava-1.png", prompt=task_prompt,
    api_url="http://localhost:8082/v1/chat/completions", model_name="lightonocr-bbox", out_prefix="lighton"
)