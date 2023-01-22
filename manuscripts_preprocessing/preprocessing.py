import cv2
import pytesseract
from paddleocr import PaddleOCR
import numpy as np


def crop_image(image, path, fname):
    """
    Crops unwanted areas from the image.
    Parameters:
        image (ndarray): The image to be cropped.
    Returns:
        image (ndarray): The cropped image.
    """
    # Since the book isn't moved during scanning, we can crop unwanted areas
    image = image[130:-130, 105:-105]
    cv2.imwrite(f"{path}/output/cropped-{fname}", image)

    return image


def mask_text(image, res, path, fname):
    """
    Uses masking with the drawn text boundaries to extract text area only.
    Parameters:
        image (ndarray): The image to be masked.
        res (list): list of boundaries of the text.
    Returns:
        image (ndarray): The masked image.
    """
    # Use masking with the drawn text boundries to extract text area only
    for line in res:
        p1, p2, p3, p4 = line[0][0], line[0][1], line[0][2], line[0][3]
        pts = [p1, p2, p3, p4]
        for i in range(4):
            pts[i] = [int(pts[i][0]), int(pts[i][1])]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))

        image = cv2.fillPoly(image, [pts], (255, 255, 255))
    cv2.imwrite(f"{path}/output/paddle-mask-{fname}", image)

    return image


def get_edges(image, th1, th2, path, fname):
    """
    Finds the edges of the image using the Canny edge detection algorithm.
    Parameters:
        image (ndarray): The image to be processed.
        th1 (int): The threshold value for the lower threshold of the algorithm.
        th2 (int): The threshold value for the upper threshold of the algorithm.
    Returns:
        edged (ndarray): The image of edges.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, th1, th2, L2gradient=True)
    cv2.imwrite(f"{path}/output/canny-{th1}-{th2}-{fname}", edged)

    return edged


def draw_bound(image, res, path, fname):
    """
    Draws lines around the boundaries of the text in the image.
    Parameters:
        image (ndarray): The image to be processed.
        res (list): list of boundaries of the text.
    Returns:
        image_bound (ndarray): The image with text boundings.
    """
    # Cast coordinates to ints in a tupples and draw lines
    image_bound = image.copy()

    # Check if image is threshed, convert it to BGR
    if len(image_bound.shape) == 2:
        image_bound = cv2.cvtColor(image_bound, cv2.COLOR_GRAY2BGR)

    for line in res:
        p1, p2, p3, p4 = line[0][0], line[0][1], line[0][2], line[0][3]
        pts = [p1, p2, p3, p4]
        for i in range(4):
            pts[i] = [int(pts[i][0]), int(pts[i][1])]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))

        image_bound = cv2.polylines(image_bound, [pts], True, (0, 0, 255), 2)

    cv2.imwrite(f"{path}/output/paddle-bound-{fname}", image_bound)
    return image_bound


def paddle_ocr(image):
    """
    Performs OCR on the image using PaddleOCR.
    Parameters:
        image (ndarray): The image to be processed.
    Returns:
        res (list): list of boundaries of the text.
    """
    ocr = PaddleOCR(
        use_angle_cls=False, rec=True, lang="en", det_db_box_thresh=0.5
    )  # need to run only once to download and load model into memory
    result = ocr.ocr(image, cls=False, rec=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            p1, p2, p3, p4 = line[0][0], line[0][1], line[0][2], line[0][3]
            # print(f"p1: {p1} p2: {p2} p3: {p3} p4: {p4}")
            pts = [p1, p2, p3, p4]
            for i in range(4):
                pts[i] = [int(pts[i][0]), int(pts[i][1])]
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))

    return res


def thresh_image(image, path, fname):
    """
    Applies thresholding on the image.
    Parameters:
        image (ndarray): The image to be processed.
    Returns:
        thresh (ndarray): The thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{path}/output/thresh-{fname}", thresh)

    return thresh


def text_opening(image, path, fname):
    """
    Applies morphological opening on the image.
    Parameters:
        image (ndarray): The image to be processed.
    Returns:
        opening (ndarray): The opened image.
    """
    opening = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (111, 111))
    cv2.imwrite(f"{path}/output/opening-{fname}", opening)

    return opening


def quantize_colors(image, path, fname):
    """
    Applies color quantization on the image.
    Parameters:
        image (ndarray): The image to be processed.
    Returns:
        result2 (ndarray): The quantized image.
    """
    Z = image.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    result = center[label.flatten()]
    result2 = result.reshape((image.shape))

    cv2.imwrite(f"{path}/output/quantization-{fname}", result2)
    return result2


def tess_ocr(image, path, fname):
    """
    Performs OCR on the image using Tesseract.
    Parameters:
        image (ndarray): The image to be processed.
    Returns:
        image (ndarray): The image with text boundings.
    """
    # Check if image is threshed, convert it to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    df = pytesseract.image_to_data(
        image, lang="hye", config="--psm 12", output_type="data.frame"
    )
    max_level = max(df["level"])

    # Drawing rectangles around recognized text based on level
    for i in range(1, len(df)):
        if df["level"][i] == max_level:
            (x, y, w, h) = (
                df["left"][i],
                df["top"][i],
                df["width"][i],
                df["height"][i],
            )
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imwrite(f"{path}/output/tess-{fname}", image)
    return image