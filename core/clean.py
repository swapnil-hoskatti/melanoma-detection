# imports - third party imports
from . import cv2

def remove_hair(img):
    # input img is numpy.ndarray
    """
        Following are the DHR tasks involved:
        -- Applying Morphological Black-Hat transformation
        -- Creating the mask for InPainting task
        -- Applying inpainting algorithm on the image
    """

    src = img

    # Convert the original image to grayscale
    gray_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    
    # adding to remove black smudges in images with too much hair
    blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_TOPHAT, kernel)

    # intensify the hair contours in preparation for the inpainting algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)

    # cv2.imwrite(dest_path, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst