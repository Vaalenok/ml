import cv2
import numpy as np
from pdf2image import convert_from_path


def pdf_to_opencv(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=r'C:\Program Files\poppler\Library\bin')
    img = np.array(pages[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def remove_small_components(binary_img, min_area=40):
    if binary_img.dtype != np.uint8:
        binary_img = binary_img.astype(np.uint8)

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8)

    if nb_components <= 1:
        return binary_img

    sizes = stats[1:, cv2.CC_STAT_AREA]
    cleaned = np.zeros(output.shape, dtype=np.uint8)

    for i in range(nb_components - 1):
        if sizes[i] >= min_area:
            cleaned[output == i + 1] = 255

    return cleaned


def preprocess_for_ocr(file_path, return_binary=False, upscale_factor=1.5, deskew=True, min_area=52):
    if file_path.lower().endswith(".pdf"):
        img = pdf_to_opencv(file_path, dpi=300)
    else:
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Масштабирование
    if upscale_factor != 1.0:
        h, w = img.shape[:2]
        new_w = int(w * upscale_factor)
        new_h = int(h * upscale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Улучшение контраста
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Шумоподавление
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)

    # Повышение резкости
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    # Выравнивание наклона
    if deskew:
        binary_for_deskew = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary_for_deskew, kernel, iterations=2)

        coords = np.column_stack(np.where(dilated > 0))

        if len(coords) > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = sharpened.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            sharpened = cv2.warpAffine(sharpened, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

    if return_binary:
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41, 12
        )

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = remove_small_components(binary, min_area=min_area)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    return sharpened


# Тесты
if __name__ == "__main__":
    path = "data/docs/сканирование (2).pdf"

    original = pdf_to_opencv(path) if path.lower().endswith(".pdf") else \
        cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

    result_gray = preprocess_for_ocr(path, return_binary=False, upscale_factor=1.5)
    result_binary = preprocess_for_ocr(path, return_binary=True, upscale_factor=1.5, min_area=15)

    cv2.imshow("Original", original)
    cv2.imshow("Preprocessed Grayscale", result_gray)
    cv2.imshow("Preprocessed Binary", result_binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
