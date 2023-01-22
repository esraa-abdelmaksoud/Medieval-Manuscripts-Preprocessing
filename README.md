# Medieval Manuscripts Preprocessing

The classification of historical documents usually includes three tasks as follows:
1) Font classification
2) Location classification
3) Date classification

In order to classify the fonts or the handwritings of manuscripts to know the author, image preprocessing would be necessary for better models training. Thus, this repository includes some preprocessing and text detection function. Futhermore, details about each process are given. The original manuscript used in this repository was posted by St. Pölten University on LinkedIn.

## Original manuscript

<img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/1669902888704.jpeg" width="40%" height="40%">

## Cropping

Since the manuscript is a part of a book, slicing the image array or cropping the image is a better option than detecting edges. This is because the 4th edge of the book where the pages are connected might not be detected because of the light. Whenever the edge detection algorithm parameters are changed, more noise will be detected instead of that edge. Thus,cropping the image is a better choice because the book is not moved during scanning.

<img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/output/cropped-1669902888704.jpeg" width="40%" height="40%">

## Thresholding vs Quantization

Usually, to have clear handwrtings for model training, it is necessary to remove the background noise. This is to make the focus of the classification model the handwriting only regardless of the background. Thus, using thresholding (binarization) to remove the background can be a good option if no darkness or shadows exist in the image. However, if there are dark stains, they may affect the binarization if the thresholds are not chosen acurately.

Two-step color quantization can also be used. This is by detecting the text and applying it to each line seperately then to the whole page. This is because using two centoids in k-means algorithm while having some drawings may result in losing some text from the manuscript image. However, there might be some text if different colors in the same line that makes binarization preferred over quantization.

<img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/output/thresh-1669902888704.jpeg" width="40%" height="40%"> <img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/output/quantization-1669902888704.jpeg" width="40%" height="40%">

## Morphological Operations



<img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/output/opening-1669902888704.jpeg" width="40%" height="40%">

## PaddleOCR vs Tesseract Text Detection

<img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/output/paddle-bound-1669902888704.jpeg" width="40%" height="40%"> <img src="https://github.com/esraa-abdelmaksoud/Medieval-Manuscripts-Preprocessing/blob/main/output/tess-1669902888704.jpeg" width="40%" height="40%">
