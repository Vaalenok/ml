import cv2
import numpy as np
from pdf2image import convert_from_path


def deskew(gray):
    h, w = gray.shape[:2]
    margin_y, margin_x = int(h * 0.03), int(w * 0.03)
    roi = gray[margin_y:h - margin_y, margin_x:w - margin_x]

    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)

    best_angle = 0
    best_score = -1

    for angle in np.arange(-5, 5.1, 0.5):
        M = cv2.getRotationMatrix2D((roi.shape[1] // 2, roi.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (roi.shape[1], roi.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        projection = np.sum(rotated, axis=1)
        score = np.var(projection)

        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.3:
        return gray

    M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remove_noise_components(binary_black_on_white, min_area=10):
    inverted = cv2.bitwise_not(binary_black_on_white)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue

        aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
        if area < 30 and aspect < 1.5:
            continue

        cleaned[output == i] = 255

    return cv2.bitwise_not(cleaned)


def assess_document(gray):
    background_mask = gray > 200
    if background_mask.sum() < 1000:
        return True, True

    background_pixels = gray[background_mask]
    bg_std = np.std(background_pixels)
    scan_mode = bg_std >= 8

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blurred)
    noise_level = np.mean(diff[background_mask])
    needs_denoise = noise_level > 3.0

    return scan_mode, needs_denoise


def fit_to_screen(img, max_w=1800, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img

    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def process_single_page(img, return_binary=False, do_deskew=True, min_area=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if do_deskew:
        gray = deskew(gray)

    scan_mode, needs_denoise = assess_document(gray)

    if needs_denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    if scan_mode:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray

    if not return_binary:
        kernel_sharp = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        result = cv2.filter2D(enhanced, -1, kernel_sharp)
        return np.clip(result, 0, 255).astype(np.uint8)

    if scan_mode:
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 67, 15)
    else:
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = remove_noise_components(binary, min_area=min_area)

    return binary


def preprocess_for_ocr(file_path, return_binary=False, do_deskew=True, min_area=10, dpi=300):
    if file_path.lower().endswith(".pdf"):
        pages_pil = convert_from_path(file_path, dpi=dpi, poppler_path=r'C:\Program Files\poppler\Library\bin')
        images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages_pil]
    else:
        img = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w < 1500:
            scale = 1500 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        images = [img]

    return [process_single_page(img, return_binary=return_binary, do_deskew=do_deskew, min_area=min_area)
            for img in images]


if __name__ == "__main__":
    path = "data/docx/сканирование0001.pdf"

    if path.lower().endswith(".pdf"):
        pages_pil = convert_from_path(path, dpi=300, poppler_path=r'C:\Program Files\poppler\Library\bin')
        originals = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages_pil]
    else:
        originals = [cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)]

    result_gray = preprocess_for_ocr(path, return_binary=False)
    result_binary = preprocess_for_ocr(path, return_binary=True)

    for i, (orig, gray, binary) in enumerate(zip(originals, result_gray, result_binary)):
        cv2.imshow(f"Original [{i + 1}]", fit_to_screen(orig))
        cv2.imshow(f"Grayscale [{i + 1}]", fit_to_screen(gray))
        cv2.imshow(f"Binary [{i + 1}]", fit_to_screen(binary))

    cv2.waitKey(0)
    cv2.destroyAllWindows()