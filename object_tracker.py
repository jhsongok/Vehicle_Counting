from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/Relaxing highway traffic.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

in_counter =[]
out_counter =[]
tot_counter = []

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    in_count = 0
    out_count = 0

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape
        cv2.line(img, (380, 400), (880, 400), (0, 255, 0), thickness=10)
        
        center_x = int(((bbox[0])+(bbox[2]))/2) 
        center_y = int(((bbox[1])+(bbox[3]))/2)

        # in count 처리
        if center_x > 380 and center_x < 600:
            if center_y > 400 and center_y < 410:
                if class_name == 'car' or class_name == 'truck':
                    in_counter.append(int(track.track_id))
                    tot_counter.append(int(track.track_id))
                    cv2.line(img, (380, 400), (880, 400), (0, 255, 255), thickness=10)
                                        
        # out count 처리
        if center_x > 685 and center_x < 880:
            if center_y > 400 and center_y < 410:
                if class_name == 'car' or class_name == 'truck':
                    out_counter.append(int(track.track_id))
                    tot_counter.append(int(track.track_id))
                    cv2.line(img, (380, 400), (880, 400), (0, 255, 255), thickness=10)
                                    
    in_count = len(set(in_counter))
    cv2.putText(img, "IN " + str(in_count), (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
    out_count = len(set(out_counter))
    cv2.putText(img, "OUT " + str(out_count), (800, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
    total_count = len(set(tot_counter))
    cv2.putText(img, "TOTAL " + str(total_count), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
    out.write(img)

vid.release()
out.release()
