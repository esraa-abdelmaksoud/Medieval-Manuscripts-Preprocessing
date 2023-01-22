from preprocessing import *
import cv2

path = input("Enter the path to folder: ")
fname = input("Enter image name including extension: ")
# path = "/mnt/E/Projects/Medieval-images-opencv/"
# fname = '1669902892924.jpeg'
# fname = "1669902888704.jpeg"

image = cv2.imread(f"{path}/{fname}")

cropped = crop_image(image, path, fname)
res = paddle_ocr(cropped)
threshed = thresh_image(cropped, path, fname)
draw_bound(threshed, res, path, fname)
get_edges(cropped, 300, 500, path, fname)
opening = text_opening(threshed, path, fname)
quantize_colors(cropped, path, fname)
tess_ocr(cropped, path, fname)