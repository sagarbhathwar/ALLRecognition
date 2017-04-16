import cv2


def supress_background(img):
    blur = cv2.GaussianBlur(img, (289, 289), 400)
    sup = img - blur
    return sup


def otsu_threshold(img):
    ret, thres = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
    return thres


def fill_holes(img):
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, B)
    return img_open
