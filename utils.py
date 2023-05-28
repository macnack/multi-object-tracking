import os
from os.path import isfile, join
from sklearn.metrics import jaccard_score
from skimage.metrics import structural_similarity as ssim_fun
import cv2 as cv
import numpy as np
from utils_cv import show2, show, get_iou, get_histogram_correlation
from scipy.optimize import linear_sum_assignment


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


def get_area(box):
    height = abs(box[1] - box[3])
    width = abs(box[0] - box[2])
    return height * width


def get_ssim(source, target):
    target_resized = cv.resize(target, dsize=(
        source.shape[1], source.shape[0]))
    # show2(source, target_resized)
    return ssim_fun(source, target_resized, data_range=source.max() - source.min(), channel_axis=2)


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


def hungarian_algorithm(matrix):
    row_idx, col_idx = linear_sum_assignment(-matrix)
    # return ID of most propability object
    return row_idx, col_idx


def bipartite_matrix(last_frame_path, frame_path, data):
    last_objects = boxes_in_frame(data[last_frame_path])
    last_frame = cv.imread(last_frame_path, cv.IMREAD_COLOR)

    current_objects = boxes_in_frame(data[frame_path])
    curr_frame = cv.imread(frame_path, cv.IMREAD_COLOR)

    adjacency_matrix = np.empty((0, len(last_objects)))
    for id_last, last in last_objects.items():
        _, last_bbox = last
        last_frame_cropped = last_frame[last_bbox[1]: last_bbox[3], last_bbox[0]: last_bbox[2], :]
        row = np.zeros(len(current_objects))
        i = 0
        for id_curr, curr in current_objects.items():
            _, curr_bbox = curr
            curr_frame_cropped = curr_frame[curr_bbox[1]: curr_bbox[3], curr_bbox[0]: curr_bbox[2], :]
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


def read_bboxes(filename, prefix):
    data = {}
    with open(filename, 'r') as f:
        while line := f.readline():
            filename = line.rsplit()[0]
            number_of_objects = f.readline().rsplit()[0]
            objects = []
            for i in range(int(float(number_of_objects))):
                [x, y, w, h] = f.readline().rsplit()
                if float(x) < 0.0:
                    x = str(0)
                if float(y) < 0.0:
                    y = str(0)
                object = [float(x), float(y), float(
                    x)+float(w), float(y)+float(h)]
                object = [int(round(x)) for x in object]
                objects.append(object)
            data[prefix+'/'+str(filename)] = objects
    return data


if __name__ == '__main__':
    path = "frames"
    images_path = load_data_paths(path)
    data = read_bboxes(filename="bboxes.txt", prefix=path)
    last_frame = None

    with open("result.txt", 'w') as output:
        output.close()

    for img_path in images_path:
        current_objects = data[img_path]
        frame = cv.imread(img_path, cv.IMREAD_COLOR)
        output = [-1 for x in range(len(current_objects))]
        # print(img_path)
        if last_frame is not None:
            last_objects = data[last_frame]
            last_frame = cv.imread(last_frame, cv.IMREAD_COLOR)

            rows = max(len(last_objects)+1, len(current_objects)+1)

            adjacency_matrix = np.ones(shape=(rows, len(current_objects))) * 0.3
            
            for id_last, last_bbox in enumerate(last_objects):
                last_object = last_frame[last_bbox[1]: last_bbox[3], last_bbox[0]: last_bbox[2], :]
                row = np.empty((len(current_objects)))
                
                for id, curr_bbox in enumerate(current_objects):
                    curr_object = frame[curr_bbox[1]: curr_bbox[3], curr_bbox[0]: curr_bbox[2], :]
                    iou = np.round(get_iou(last_bbox, curr_bbox), 2)
                    ssim = get_ssim(last_object, curr_object)
                    hist = get_histogram_correlation(last_object, curr_object)
                    adjacency_matrix[id_last, id] = combine_probabilities([iou, ssim, hist], [0.1, 0.3, 0.6])
            
            idx_row, idx_col = hungarian_algorithm(adjacency_matrix.T)
            output = []
            for j, i in zip(idx_col, idx_row):
                if j >= len(last_objects):
                    j = -1
                output.append(j)
            # print(adjacency_matrix)
        output_buffer = " ".join(map(str, output)) + '\n'
        output_buffer_file = "\n".join(map(str, output)) + '\n'
        print(output_buffer)
        with open("result.txt", 'a') as output_file:
            output_file.write(f"{img_path[len(path)+1:]}\n")
            output_file.write(f"{len(current_objects)}\n")
            output_file.write(output_buffer_file)
        last_frame = img_path

    # data = read_lines("labels/0000.txt")
    # adjacency_matrix = bipartite_matrix(images_path[0], images_path[1], data)
    # print(adjacency_matrix)
