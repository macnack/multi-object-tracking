import os
from os.path import isfile, join
from sklearn.metrics import jaccard_score
from skimage.metrics import structural_similarity as ssim_fun
import cv2 as cv
import numpy as np
from utils_cv import show2, show


def load_data_paths(dataset_path):
    images = [join(dataset_path, f) for f in os.listdir(
        dataset_path) if isfile(join(dataset_path, f)) and not f.startswith("._")]
    return sorted(images)


def read_lines(dataset):
    content = {}
    objects = {}
    frame_idx = 0
    with open(dataset, 'r') as f:
        for line in f:
            line = line.strip().split()
            frame, id, type, _, _, _, left, top, right, bottom, h, w, l, _, _, _, _ = line
            if int(id) == -1:
                continue
            if int(frame) != frame_idx:
                if objects:
                    content[f"training/0000/{frame_idx:06d}.png"] = objects
                frame_idx = int(frame)
                objects = {}
            objects[id] = (type, [(int(float(left)), int(float(top))),
                           (int(float(right)), int(float(bottom)))])
        if objects:  # Add objects from the last frame
            content[f"training/0000/{frame_idx:06d}.png"] = objects
    return content


def get_jac(img_source, image_target):
    img_source_1D = np.array(img_source).ravel()
    image_target_1D = np.array(image_target).ravel()
    return jaccard_score(img_source_1D, image_target_1D, average='micro')


def show_bbox(data):
    for key, value in data.items():
        img = cv.imread(key, cv.IMREAD_COLOR)
        last_ids = []
        ids = []
        for id, obj in value.items():
            ids.append(id)
            type = obj[0]
            pt1, pt2 = obj[1]
            color = (0, 255, 0)
            if type in ['Car', 'Van']:
                color = (255, 0, 0)
            if type in ['Cyclist', 'Pedestrian']:
                colot = (0, 0, 255)
            cv.rectangle(img, pt1, pt2, color, 3)
            cv.putText(img, f'{id}', pt1, cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2, cv.LINE_AA)
        for id in ids:
            if id not in last_ids:
                count = last_ids.count(id)
                print("-1 " * count, end="")
        last_ids = ids
        cv.imshow(key, img)
        cv.waitKey(200)
        cv.destroyAllWindows()


def boxes_in_frame(frame, draw=None):
    boxes = {}
    for id, obj in frame.items():
        type = obj[0]
        pt1, pt2 = obj[1]
        if draw is not None:
            color = (0, 255, 0)
            if type in ['Car', 'Van']:
                color = (255, 0, 0)
            if type in ['Cyclist', 'Pedestrian']:
                color = (0, 0, 255)
            cv.rectangle(draw, pt1, pt2, color, 3)
            cv.putText(draw, f'{id}', pt1, cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2, cv.LINE_AA)
        boxes[int(id)] = (type, [pt1[0], pt1[1], pt2[0], pt2[1]])
    return boxes


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


def get_area(box):
    height = abs(box[1] - box[3])
    width = abs(box[0] - box[2])
    return height * width


def get_ssim(source, target):
    target_resized = cv.resize(target, (source.shape[1], source.shape[0]))
    # show2(source, target_resized)
    return ssim_fun(source, target_resized, data_range=source.max() - source.min(), channel_axis=2)


def get_histogram_correlation(source, target):
    # source = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    # target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    # # target = cv.resize(target, (source.shape[1], source.shape[0], source.shape[3]))
    # source_hist = cv.calcHist([source], [0], None, [255], [0, 256])
    # target_hist = cv.calcHist([target], [0], None, [256], [0, 256])
    # source_hist = cv.normalize(source_hist, source_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    # target_hist = cv.normalize(target_hist, target_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    show(source, "source")
    source = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    show(source, "source")
    source_hist = cv.calcHist([source], [0], None, [255], [0, 256])
    # correlation = cv.compareHist(source_hist, target_hist, cv.HISTCMP_CORREL)
    return 1.0


def get_center_box(box):
    height = abs(box[1] - box[3])
    width = abs(box[0] - box[2])
    return [box[0]+width/2.0, box[1]+height/2.0]


def distance_to_probability(bbox1, bbox2, decay_rate=0.1):
    bbox1_center = np.array(get_center_box(bbox1))
    bbox2_center = np.array(get_center_box(bbox2))
    distance = np.linalg.norm(bbox2_center - bbox1_center)
    probability = np.exp(-decay_rate * distance)
    return probability


def normalize_to_one(value):
    return value / abs(value)


def combine_probabilities(probabilities, weights=None):
    if weights is None:
        weights = [1.0 / len(probabilities) for _ in range(len(probabilities))]
    else:
        if len(weights) != len(probabilities) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Invalid weights! The number of weights should match the number of probabilities, "
                             "and the sum of weights should be equal to 1.0.")

    combined_prob = sum(p * w for p, w in zip(probabilities, weights))
    return combined_prob


if __name__ == '__main__':
    images_path = load_data_paths('training/0000')
    data = read_lines("labels/0000.txt")
    img_path = images_path[1]

    img = cv.imread(img_path, cv.IMREAD_COLOR)

    frame_last = cv.imread(images_path[0], cv.IMREAD_COLOR)
    frame = cv.imread(images_path[1], cv.IMREAD_COLOR)

    last_objects = boxes_in_frame(data[images_path[0]])
    current_objects = boxes_in_frame(data[images_path[1]])
    adjacency_matrix = np.empty((0, len(current_objects)))
    # for id_current, curr in current_objects.items():
    #     _, pts_curr = curr
    #     output = ""
    #     col = np.zeros(shape=(len(last_objects),1))
    #     j = 0
    #     frame_cropped = frame[pts_curr[1]: pts_curr[3], pts_curr[0]: pts_curr[2], :]
    #     for id_last, last in last_objects.items():
    #         _, pts_last = last
    #         iou = get_iou(pts_last, pts_curr)
    #         iou = round(iou, 2)
    #         frame_last_cropped = frame_last[pts_last[1]: pts_last[3], pts_last[0]: pts_last[2], :]
    #         jac = get_ssim(frame_last_cropped, frame_cropped)
    #         dist = distance_to_probability(pts_last, pts_curr)
    #         dist = round(dist,2)
    #         comb = combine_probabilities([jac, iou])
    #         output += "IOU id: " + \
    #             str(id_current) + " id_last: " + \
    #             str(id_last) + " " + str(iou) + '\n'
    #         output += "SSIM id: " + \
    #             str(id_current) + " id_last: " + \
    #             str(id_last) + " " + str(jac) + '\n'
    #         output += "Distance id: " + \
    #             str(id_current) + " id_last: " + \
    #             str(id_last) + " " + str(dist) + '\n'
    #         output += "Weig id: " + str(id_current) + " id_last: " + str(id_last) + " " + str(comb) +'\n\n'
    #         col[j] = iou
    #         j+=1
    #     print(col)
    # print(output)
    curr_frame = cv.imread(images_path[1], cv.IMREAD_COLOR)
    last_frame = cv.imread(images_path[0], cv.IMREAD_COLOR)
    adjacency_matrix = np.empty((0, len(last_objects)))
    for id_last, last in last_objects.items():
        _, last_bbox = last
        last_frame_cropped = last_frame[last_bbox[1]
            : last_bbox[3], last_bbox[0]: last_bbox[2], :]
        row = np.zeros(len(current_objects))
        i = 0
        for id_curr, curr in current_objects.items():
            _, curr_bbox = curr
            curr_frame_cropped = curr_frame[curr_bbox[1]
                : curr_bbox[3], curr_bbox[0]: curr_bbox[2], :]
            iou = np.round(get_iou(last_bbox, curr_bbox), 2)
            ssim = get_ssim(last_frame_cropped, curr_frame_cropped)
            dist = distance_to_probability(last_bbox, curr_bbox)
            dist = np.round(dist, 2)
            hist = get_histogram_correlation(
                last_frame_cropped, curr_frame_cropped)
            row[i] = hist
            i += 1
        adjacency_matrix = np.vstack((adjacency_matrix, row))

    print(adjacency_matrix)
    show2(last_frame, curr_frame)
    # print(get_iou(pts_last, pts))
    # cv.rectangle(img, (pts_last[0], pts_last[1]), (pts_last[2], pts_last[2]), color=(255,0,0), thickness=3)
    # cv.rectangle(img, (pts[0], pts[1]), (pts[2], pts[2]), color=(0,255,0), thickness=3)
    # cv.imshow(img_path[0], img)
    # cv.waitKey(10000)
    # get_iou(bb0, bb1)

    # print(data[images_path[0]])
    # print()
    # print(data[images_path[1]])
    # for obj in data[images_path[0]]:
    #     print(obj)
    # cv.waitKey(1000)
