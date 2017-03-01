from read import read_data

import cv2
import numpy as np

from visualize import show_img, compare_images
from visualize import mark_cancerous_lymphocytes

from segment import detect_edges, get_mask


if __name__ == "__main__":
    IMG_PATH = "ALL_IDB1\\im\\"
    XYC_PATH = "ALL_IDB1\\xyc\\"

    # This dataframe stores id, image path, bool for presence of blasts and
    # co-ordinates of blasts if any: id, img_path, has_blasts, blast_xy
    df = read_data(IMG_PATH, XYC_PATH)
    img = cv2.imread(df.iloc[1].img_path)
    mask = get_mask(img)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    mark_cancerous_lymphocytes(img, df.iloc[1].blast_xy)
    # edges = detect_edges(img, 100, 147)
    compare_images(img, segmented_img)

