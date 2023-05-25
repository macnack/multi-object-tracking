import os
from os.path import isfile, join
import argparse
import cv2 as cv

def load_data_paths(dataset_path):
    images = [join(dataset_path, f) for f in os.listdir(
        dataset_path) if isfile(join(dataset_path, f))]
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
            objects[id] = (type, [(int(float(left)), int(float(top))), (int(float(right)), int(float(bottom)))])
        if objects:  # Add objects from the last frame
            content[f"training/0000/{frame_idx:06d}.png"] = objects
    return content

if __name__ == '__main__':
    data = read_lines("labels/0000.txt")
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
            cv.rectangle(img, pt1, pt2,color,3)
            cv.putText(img,f'{id}',pt1, cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
        for id in ids:
            if id not in last_ids:
                count = last_ids.count(id)
                print("-1 " * count, end="")
        last_ids = ids
        cv.imshow(key, img)
        cv.waitKey(200)
        cv.destroyAllWindows()
    # for obj in data[images_path[0]]:
    #     print(obj)
    # cv.waitKey(1000)