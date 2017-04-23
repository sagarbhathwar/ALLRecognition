import cv2
import numpy as np
from skimage import measure
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from visualize import compare_images


def fill_holes(img):
    im = np.copy(img)
    # Find contours in image. Retrieves all of the contours and organizes
    # them into a two-level hierarchy. At the top level, there are external
    # boundaries of the components. At the second level, there are boundaries
    # of the holes. If there is another contour inside a hole of a connected
    # component, it is still put at the top level.
    _, contours, _ = cv2.findContours(im, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # Using the obtained contours, fill in the contours. 'thickness=-1' does
    # the filling
    for contour in contours:
        cv2.drawContours(im, contours=[contour], contourIdx=0, color=255,
                         thickness=-1)

    return im


def remove_small_cells(img):
    # Find unique connected components in the image and label each pixel as
    # belonging to a particular connected component
    # 0 will be the background
    labels = measure.label(img, neighbors=4)

    if len(np.unique(labels)) > 2000:
        med = cv2.medianBlur(img, 5)
        labels = measure.label(med)

    # This image finally contains all cells other than those with area less
    # than some specified threshold
    small_cells_eleminated_img = np.zeros(img.shape, np.uint8)

    # For each connected component, find number of pixels in it i.e. area of
    # it. If it is less than 1/600 of the image, remove it
    cell_size_threshold = int((img.shape[0] * img.shape[1]) / 1000)
    for label in np.unique(labels)[1:]:
        label_mask = np.zeros(img.shape, np.uint8)
        label_mask[labels == label] = 255
        num_fg_pixels = cv2.countNonZero(label_mask)
        if num_fg_pixels > cell_size_threshold:
            small_cells_eleminated_img = cv2.add(
                small_cells_eleminated_img, label_mask)

    return small_cells_eleminated_img


def cluster(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the three channels. We will need only a* and b* eventually for
    # clustering
    L_star, a_star, b_star = cv2.split(lab)

    # Combine only a_star and b_star color space for clustering purpose
    a_b_star = np.array((a_star.ravel(), b_star.ravel()), np.uint8).transpose()

    # Using a Gaussian Mixture Model clustering method, cluster the points in
    #  the image. Each pixel either belongs to background  (0),
    # or White Blood Cell(1) ro Red Blood Cell(2)
    model = GaussianMixture(n_components=3, max_iter=20)
    model.fit(a_b_star)
    pred = model.predict(a_b_star)

    # Reshae the predicted values into a matrix so that all pixels classified
    #  as 1 can be used for generating mask
    pred = np.reshape(pred, (img.shape[:2]))
    z = np.zeros(img.shape[:2], np.uint8)
    z[pred == 1] = 255

    return z


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
    blur = cv2.GaussianBlur(dbl, (241, 241), 40)
    sub = np.subtract(dbl, blur)
    # Contrast stretching
    cs = rescale_intensity(sub,
                           in_range=(np.percentile(sub.ravel(), 1),
                                     np.percentile(sub.ravel(), 99)),
                           out_range=(0.0, 1.0))

    """"""
    compare_images(dbl, cs)

    # cv2.imshow("Window", cs), cv2.waitKey(0), cv2.destroyAllWindows()

    thresh = threshold_otsu(cs)
    binary = cs > thresh
    binary = 1.0 - binary

    """"""
    compare_images(cs, binary)

    # Convert the image back to 0-255 range for futher processing. An open
    # operation is optional to disjoin cells.
    # Median filter can then be applied so that small speckled noise is
    # eleminated. But that joins cells, which is not ideal. So, it can be
    # deferred for now
    binary = np.array(np.multiply(binary, 255), np.uint8)
    # cv2.imshow("Window", binary), cv2.waitKey(0), cv2.destroyAllWindows()

    holes_filled = fill_holes(binary)
    cv2.imwrite("holes_filled.jpg", holes_filled)

    """"""
    compare_images(binary, holes_filled)
    # Calculate mean cell area. We sum up all the pixels in the image and
    # divide it by 510 (255 * 2) i.e. half the mean cell area in terms of
    # number of pixels
    labels = measure.label(holes_filled, neighbors=4)
    mean_cell_area = sum(holes_filled.ravel()) // len(
        np.unique(labels))
    threshold = mean_cell_area // 510

    # Color based clustering using Gaussian Mixture Model
    # Final image that contains only white blood cells
    # 0 is BG, 1 is WBC and 2 is RBC
    clustered_img = cluster(img)

    """"""
    compare_images(holes_filled, clustered_img)
    final_image = np.zeros(binary.shape, np.uint8)

    labels = measure.label(clustered_img)
    print(len(np.unique(labels)))
    for label in np.unique(labels)[1:]:
        label_mask = np.zeros(clustered_img.shape, np.uint8)
        label_mask[labels == label] = 255
        num_white_pixels = cv2.countNonZero(label_mask)
        if num_white_pixels > threshold:
            final_image = cv2.add(final_image, label_mask)

    # Combine the results from the two methods to obtain a final mask
    final_image = np.bitwise_and(final_image, holes_filled)
    small_cells_eliminated = remove_small_cells(final_image)
    small_cells_eliminated = fill_holes(small_cells_eliminated)

    """"""
    compare_images(clustered_img, small_cells_eliminated)

    return small_cells_eliminated
