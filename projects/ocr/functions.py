import cv2
import numpy as np
from pdf2image import convert_from_path


def deskew(gray):
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


def fit_to_screen(img, max_w=1800, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def preprocess_for_ocr(file_path, do_deskew=True, dpi=300):
    if file_path.lower().endswith(".pdf"):
        pages_pil = convert_from_path(file_path, dpi=dpi,
                                      poppler_path=r'C:\Program Files\poppler\Library\bin')
        images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages_pil]
    else:
        img = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        images = [img]

    result = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if do_deskew:
            gray = deskew(gray)
        result.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    return result


if __name__ == "__main__":
    # path = "data/docx/сканированиеПОВЕРНУТЫЙ.pdf"
    path = "data/img/snils-krivoi.jpeg"

    if path.lower().endswith(".pdf"):
        pages_pil = convert_from_path(path, dpi=300, poppler_path=r'C:\Program Files\poppler\Library\bin')
        originals = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages_pil]
    else:
        originals = [cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)]

    processed = preprocess_for_ocr(path)

    for i, (orig, proc) in enumerate(zip(originals, processed)):
        cv2.imshow(f"Original [{i + 1}]", fit_to_screen(orig))
        cv2.imshow(f"Processed [{i + 1}]", fit_to_screen(proc))

    cv2.waitKey(0)
    cv2.destroyAllWindows()