import os

import cv2
import matplotlib.pyplot as plt


def show_img(img, win="Window"):
    cv2.imshow("Window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mark_cancerous_lymphocytes(img, points):
    for pt in points:
        cv2.drawMarker(img, pt, (255, 255, 0), cv2.MARKER_TILTED_CROSS,
                       thickness=3, line_type=cv2.LINE_AA)


def compare_images(img1, img2, img1_label='Image 1', img2_label='Image 2'):
    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title(img1_label), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title(img2_label), plt.xticks([]), plt.yticks([])

    if os.name == 'nt':
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    plt.show()
