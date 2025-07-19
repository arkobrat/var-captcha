# VarCAPTCHA
A CAPTCHA recognition system to accurately recognize variable-length CAPTCHAs.

## Approach
This CAPTCHA recognition system was developed by tokenizing CAPTCHA images to segment them into their constituent characters and then training a convolutional neural network (CNN) to recognize the characters.

## Pre-processing
* Converted the image to grayscale for simpler analysis.
* Enhanced contrast using histogram equalization.
* Applied Gaussian adaptive thresholding for binarization.
* Used median filtering to remove noise and thin lines.
* Performed morphological opening (erosion followed by dilation) to clean the image.
* Detected all contours in the processed binary image.
* Calculate bounding boxes to locate the text region.

## Segmentation
* Region-based
    * Searched for contours within the CAPTCHA area to identify potential character regions.
    * Sorted the bounding boxes by x-coordinates to order the CAPTCHA left-to-right.
    * Set the overlap parameter to 70% to group overlapping bounding boxes for overlapping/close characters.
* Colour-based
    * Converted the CAPTCHA area to the HSV colour space to analyze hue transitions.
    * Created a histogram of the hue channel and filtered regions of the image based on hue peaks detected.

## Model
A convolutional neural network (CNN) was used as the CAPTCHA recognition model to accurately identify alphanumeric characters.

* Input dimensionality: Standardized dimensions were used in the input layers for matrix operations and weight calculations.
* One-hot encoding: Labels undergo categorical encoding to n-dimensional binary vectors (n=36 for alphanumeric characters). This prevents ordinal bias in softmax outputs and enables proper cross-entropy loss calculation.
* Augmentation pipeline: Applying transformations generates synthetic training samples, increasing dataset variance and reducing overfitting by introducing position-invariant features.

## Results
* The segmentation algorithm was able to successfully tokenize 7200 of the 8011 images in the cleaned dataset (i.e, the number of characters returned by the algorithm was the same as the number of labelled characters). The remaining 811 images were left out of the training pipeline.
* The trained model achieved a training loss of 0.0089, a validation loss of 0.0144 and a validation accuracy of 91.33%.
* Further statistics (refer to `evaluator.ipynb`):
    * Precision: 91.92%
    * Recall: 91.34%
    * F1 Score: 91.51%
