import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import img2pdf


def deskew(gray, target_dim=4000):
    h, w = gray.shape[:2]

    scale = target_dim / max(h, w)

    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)

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


def preprocess_for_ocr(file_path, output_dir="data/processed", do_deskew=True, dpi=300):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"proc_{file_name_without_ext}.pdf"
    final_output_path = os.path.join(output_dir, output_filename)

    if file_path.lower().endswith(".pdf"):
        pages_pil = convert_from_path(file_path, dpi=dpi, poppler_path=r'C:\Program Files\poppler\Library\bin')
        images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages_pil]
    else:
        images = [cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_COLOR)]

    processed_bytes_list = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if do_deskew:
            gray = deskew(gray)

        final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _, buffer = cv2.imencode(".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_bytes_list.append(buffer.tobytes())

    try:
        with open(final_output_path, "wb") as f:
            f.write(img2pdf.convert(processed_bytes_list))

        print(f"Файл успешно сохранен: {final_output_path}")
        return final_output_path
    except Exception as e:
        print(f"Ошибка при сохранении PDF: {e}")
        raise e
