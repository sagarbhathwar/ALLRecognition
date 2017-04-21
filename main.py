import os

import cv2

from read import read_data
from segment import segment_leukocytes
from visualize import compare_images, mark_cancerous_lymphocytes

if __name__ == "__main__":
    IMG_PATH = os.path.join("ALL_IDB1", "im")
    XYC_PATH = os.path.join("ALL_IDB1", "xyc")

    # This dataframe stores id, image path, bool for presence of blasts and
    # co-ordinates of blasts if any: id, img_path, has_blasts, blast_xy
    df = read_data(IMG_PATH, XYC_PATH)

    img = cv2.imread(df.loc[0].img_path)

    segmented_image = segment_leukocytes(img)
    mark_cancerous_lymphocytes(img, df.loc[0].blast_xy)
    compare_images(img, segmented_image)
