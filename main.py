from os import listdir
from os.path import isfile, join
from skimage.metrics import structural_similarity as ssim_fun
import cv2 as cv
import numpy as np
from utils_cv import get_iou, get_histogram_correlation
from scipy.optimize import linear_sum_assignment
import argparse


def load_data_paths(dataset_path):
    images = [join(dataset_path, f) for f in listdir(
        dataset_path) if isfile(join(dataset_path, f)) and not f.startswith("._")]
    return sorted(images)


def get_ssim(source, target):
    # Structural Similarity
    target_resized = cv.resize(target, dsize=(
        source.shape[1], source.shape[0]))
    return ssim_fun(source, target_resized, data_range=source.max() - source.min(), channel_axis=2)


def get_center_box(box):
    height = abs(box[1] - box[3])
    width = abs(box[0] - box[2])
    return [box[0]+width/2.0, box[1]+height/2.0]


def distance_to_probability(bbox1, bbox2, decay_rate=0.1):
    # Distance probability but not used
    bbox1_center = np.array(get_center_box(bbox1))
    bbox2_center = np.array(get_center_box(bbox2))
    distance = np.linalg.norm(bbox2_center - bbox1_center)
    probability = np.exp(-decay_rate * distance)
    return probability


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
    _, col_idx = linear_sum_assignment(-matrix)
    # return ID of most propability object
    return col_idx


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


def bipartite_graph(images_path, data, result=None):
    # Init
    last_frame = None

    # To write an output to txt
    if result is not None:
        with open(result, 'w') as output_file:
            output_file.close()

    # creating bibartitle_graph
    for img_path in images_path[:3]:
        current_objects = data[img_path]
        frame = cv.imread(img_path, cv.IMREAD_COLOR)
        output = []
        if last_frame is not None:
            last_objects = data[last_frame]
            last_frame = cv.imread(last_frame, cv.IMREAD_COLOR)

            rows = max(len(last_objects)+1, len(current_objects)+1)
            # init adjecency matrix
            adjacency_matrix = np.ones(
                shape=(rows, len(current_objects))) * 0.3
            for id_last, last_bbox in enumerate(last_objects):
                # Cropping image
                last_object = last_frame[last_bbox[1]: last_bbox[3],
                                         last_bbox[0]: last_bbox[2], :]

                for id, curr_bbox in enumerate(current_objects):
                    # Cropping image
                    curr_object = frame[curr_bbox[1]: curr_bbox[3],
                                        curr_bbox[0]: curr_bbox[2], :]

                    iou = np.round(get_iou(last_bbox, curr_bbox), 2)
                    ssim = get_ssim(last_object, curr_object)
                    hist = get_histogram_correlation(last_object, curr_object)
                    adjacency_matrix[id_last, id] = combine_probabilities(
                        [iou, ssim, hist], [0.1, 0.3, 0.6])

            # Optimaze, using hungarian algorithm
            idx_col = hungarian_algorithm(adjacency_matrix.T)

            # Create output buffer

            for j in idx_col:
                if j >= len(last_objects):
                    j = -1
                output.append(j)
        if last_frame is None:
            for _ in data[img_path]:
                output.append(-1)
        # Print output buffer
        output_buffer = " ".join(map(str, output)) + '\n'
        print(output_buffer)

        # Make history
        last_frame = img_path

        # Easy to compare format with ground_truth.txt
        if result is not None:
            output_buffer_file = "\n".join(map(str, output)) + '\n'
            with open(result, 'a') as output_file:
                output_file.write(f"{img_path[len(path)+1:]}\n")
                output_file.write(f"{len(current_objects)}\n")
                output_file.write(output_buffer_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process files in a directory and write to an output file")
    parser.add_argument("directory", nargs="?", default='frames',
                        help="path to the directory of files")
    # parser.add_argument("output_file", nargs="?",
    #                     default='out.csv', help="path to the output file")
    args = parser.parse_args()
    path = args.directory
    images_path = load_data_paths(path)
    data = read_bboxes(
        filename="/".join((path, "..", "bboxes.txt")), prefix=path)
    bipartite_graph(images_path, data, result=None)
