import cv2
import numpy as np

from skimage.color import rgb2hed, rgb2gray
from skimage import img_as_uint
from visualize import compare_images, show_img


def detect_edges(img, t_min=50, t_max=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray, t_min, t_max)
    return edge_img


def get_mask(img):
    hed = rgb2hed(img)
    gray = img_as_uint(rgb2gray(hed))
    smooth = cv2.medianBlur(np.array(gray, dtype=np.uint8), 3)
    val, thres = cv2.threshold(smooth, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres = cv2.medianBlur(thres, 9)
    return thres


def segment_cancerous_cells(img):
    return None
