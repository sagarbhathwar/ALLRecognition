import cv2
import numpy as np

from skimage import img_as_uint
from skimage.color import rgb2hed, rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu


def detect_edges(img, t_min=50, t_max=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray, t_min, t_max)
    return edge_img


def get_mask(img):
    hed = rgb2hed(img)
    gray = img_as_uint(rgb2gray(hed))
    smooth = cv2.GaussianBlur(np.array(gray, dtype=np.uint8), (13,13), 40)
    val, thres = cv2.threshold(smooth, 50, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    blur = cv2.medianBlur(thres, 5)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel=kernel,
                               iterations=4)
    return opening


def segment_leukocytes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Working in double precision gives an advantage of getting a continuous
    # histogram while doing contrast stretching, which, then can be
    # interpolated to 0-255 easily
    dbl = np.array(np.divide(gray, 255), np.float64)

    # Use Gaussian filter to construct the background and then subtract it
    # from the foreground. Doing this would give a bimodal histogram when
    # contrast stretched so that OSTU tresholding method gives a find
    # threshold, separating background and foreground. We then invert the
    # background and foreground by subtracting the image by 1.0
    blur = cv2.GaussianBlur(dbl, (161, 161), 40)
    sub = np.subtract(dbl, blur)
    # Contrast stretching
    cs = rescale_intensity(sub,
                           in_range=(np.percentile(sub.ravel(), 1),
                                     np.percentile(sub.ravel(), 99)),
                           out_range=(0.0, 1.0))
    thresh = threshold_otsu(cs)
    binary = cs > thresh
    binary = 1.0 - binary

    # Convert the image back to 0-255 range for futher processing. An open
    # operation is optional to disjoin cells.
    # Median filter can then be applied so that small speckled noise is
    # eleminated. But that joins cells, which is not ideal. So, it can be
    # deferred for now
    binary = np.array(np.multiply(binary, 255), np.uint8)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, op=cv2.MORPH_OPEN, kernel=kernel,
                               iterations=2)
    return opening

