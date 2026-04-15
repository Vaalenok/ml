import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import base64
import img2pdf


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