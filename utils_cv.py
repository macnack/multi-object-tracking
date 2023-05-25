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
