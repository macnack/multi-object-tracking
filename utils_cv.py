from os.path import isfile, join
import os
import numpy as np
import cv2 as cv


images_dir = "dane/data"

def load_images():
    images = [join(images_dir, f) for f in os.listdir(
        images_dir) if isfile(join(images_dir, f))]
    return sorted(images)

def read_img(path, scale):
    img = cv.imread(path, cv.IMREAD_COLOR)
    img = cv.resize(img, None, fx=scale, fy=scale)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray

def pts_to_corners(pts):
    left = np.min(pts[:, 0])
    right = np.max(pts[:, 0])
    top = np.min(pts[:, 1])
    bottom = np.max(pts[:, 1])
    return left, top, right, bottom


def show_images_in_grid(images):
    idx = 0
    output = np.array([])
    output_shape = (100, 100)
    for row in range(int(len(images) / 4)):
        row_ = np.array([])
        for col in range(5):
            if idx < len(images):
                temp = images[idx]  # cv.imread(images[idx], cv.IMREAD_COLOR)
                temp = cv.resize(temp, dsize=output_shape)
                if row_.size == 0:
                    row_ = temp
                else:
                    row_ = np.concatenate((row_, temp), axis=1)
            else:
                row_ = np.concatenate(
                    (row_, np.zeros(temp.shape, dtype=np.uint8)), axis=1)
            idx += 1
        if output.size == 0:
            output = row_
        else:
            output = np.concatenate((output, row_), axis=0)
    return output


def show(img, title="One Image"):
    while True:
        cv.imshow(title, img)
        cv.waitKey(300)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

def show2(img1, img2, title="Two images"):
    if img1.ndim != img2.ndim:
        if img1.ndim > img2.ndim:
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        else:
            img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    output = np.concatenate((img1, img2), axis=1)
    show(output, title)

def get_histogram_correlation(source, target):
    target_resized = cv.resize(target, dsize=(source.shape[1], source.shape[0]))
    source_hist = cv.cvtColor(source, cv.COLOR_BGR2HSV)
    target_hist = cv.cvtColor(target_resized, cv.COLOR_BGR2HSV)

    histSize = [50, 60]
    ranges = [0, 180, 0, 256]
    channels = [0, 1]
    source_hist = cv.calcHist([source_hist], channels=channels, mask=None, histSize=histSize, ranges=ranges, accumulate=False)
    cv.normalize(source_hist, source_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    target_hist = cv.calcHist([target_hist], channels=channels, mask=None, histSize=histSize, ranges=ranges, accumulate=False)
    cv.normalize(target_hist, target_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    return cv.compareHist(source_hist, target_hist, cv.HISTCMP_CORREL)

def get_iou(boxA, boxB):
    # Calculate coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate the Intersection over Union
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou