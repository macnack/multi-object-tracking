import cv2 as cv

def show(img, title="One Image"):
    while True:
        cv.imshow(title, img)
        cv.waitKey(300)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def get_histogram_correlation(source, target):
    target_resized = cv.resize(target, dsize=(
        source.shape[1], source.shape[0]))
    source_hist = cv.cvtColor(source, cv.COLOR_BGR2HSV)
    target_hist = cv.cvtColor(target_resized, cv.COLOR_BGR2HSV)

    histSize = [50, 60]
    ranges = [0, 180, 0, 256]
    channels = [0, 1]
    source_hist = cv.calcHist([source_hist], channels=channels,
                              mask=None, histSize=histSize, ranges=ranges, accumulate=False)
    cv.normalize(source_hist, source_hist, alpha=0,
                 beta=1, norm_type=cv.NORM_MINMAX)

    target_hist = cv.calcHist([target_hist], channels=channels,
                              mask=None, histSize=histSize, ranges=ranges, accumulate=False)
    cv.normalize(target_hist, target_hist, alpha=0,
                 beta=1, norm_type=cv.NORM_MINMAX)

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
