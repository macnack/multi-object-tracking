from sklearn.metrics import jaccard_score
from utils import get_iou, get_ssim, distance_to_probability, get_histogram_correlation
import numpy as np
import cv2 as cv

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

def bipartite_matrix(last_frame_path, frame_path, data):
    last_objects = boxes_in_frame(data[last_frame_path])
    last_frame = cv.imread(last_frame_path, cv.IMREAD_COLOR)

    current_objects = boxes_in_frame(data[frame_path])
    curr_frame = cv.imread(frame_path, cv.IMREAD_COLOR)

    adjacency_matrix = np.empty((0, len(last_objects)))
    for id_last, last in last_objects.items():
        _, last_bbox = last
        last_frame_cropped = last_frame[last_bbox[1]: last_bbox[3],
                                        last_bbox[0]: last_bbox[2], :]
        row = np.zeros(len(current_objects))
        i = 0
        for id_curr, curr in current_objects.items():
            _, curr_bbox = curr
            curr_frame_cropped = curr_frame[curr_bbox[1]: curr_bbox[3],
                                            curr_bbox[0]: curr_bbox[2], :]
            iou = np.round(get_iou(last_bbox, curr_bbox), 2)
            ssim = get_ssim(last_frame_cropped, curr_frame_cropped)
            dist = distance_to_probability(last_bbox, curr_bbox)
            dist = np.round(dist, 2)
            hist = get_histogram_correlation(
                last_frame_cropped, curr_frame_cropped)
            row[i] = hist
            i += 1
        adjacency_matrix = np.vstack((adjacency_matrix, row))
    return adjacency_matrix

if __name__ == "__main__":
    data = read_lines("labels/0000.txt")