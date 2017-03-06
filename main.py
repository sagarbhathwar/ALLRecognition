import os

import cv2
import numpy as np

from read import read_data
from visualize import show_img, compare_images
from visualize import mark_cancerous_lymphocytes
from segment import detect_edges, get_mask


if __name__ == "__main__":
    IMG_PATH = os.path.join("ALL_IDB1", "im")
    XYC_PATH = os.path.join("ALL_IDB1","xyc")

    # This dataframe stores id, image path, bool for presence of blasts and
    # co-ordinates of blasts if any: id, img_path, has_blasts, blast_xy
    df = read_data(IMG_PATH, XYC_PATH)
    img_data = df.iloc[0]
    img = cv2.imread(img_data['img_path'])
    mask = get_mask(img)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    mark_cancerous_lymphocytes(img, img_data['blast_xy'])
    compare_images(img, segmented_img)

