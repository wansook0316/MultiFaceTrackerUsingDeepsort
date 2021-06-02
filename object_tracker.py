import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS

from mtcnn import MTCNN 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

"""
python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video 0 \
    --weights ./weights/yolov3-wider_16000.tf \
    --num_classes 1 \
    --output_format MP4V \
    --output ./resources/video/out/myface.mp4 \

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/2.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/2 \
    --output ./resources/video/out/2.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/T-ara.mov \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/T-ara \
    --output ./resources/video/out/T-ara.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/T-ara_pred.txt

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/BrunoMars.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/BrunoMars \
    --output ./resources/video/out/BrunoMars.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/BrunoMars_pred.txt

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/Darling.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/Darling \
    --output ./resources/video/out/Darling.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/Darling_pred.txt

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/GirlsAloud.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/GirlsAloud \
    --output ./resources/video/out/GirlsAloud.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/GirlsAloud_pred.txt

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/HelloBubble.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/HelloBubble \
    --output ./resources/video/out/HelloBubble.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/HelloBubble_pred.txt

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/Westlife.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/Westlife \
    --output ./resources/video/out/Westlife.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/Westlife_pred.txt

python object_tracker.py \
    --classes ./model_data/labels/widerface.names \
    --video ./resources/video/in/Apink.mp4 \
    --weights ./weights/yolov3-wider_16000.tf \
    --output_format MP4V \
    --database ./resources/database/Apink \
    --output ./resources/video/out/Apink.mp4 \
    --num_classes 1 \
    --max_face_threshold 0.6871912959056619 \
    --eval ./resources/gt/Apink_pred.txt
"""


flags.DEFINE_string('classes', './model_data/labels/widerface.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3-wider_16000.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './resources/video/in/1.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('database', './resources/database/1',
                    'path to database file for identification)')
flags.DEFINE_string('output', './resources/video/out/1.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'MP4V', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_float('max_face_threshold', 0.6871912959056619, 'face threshold')
flags.DEFINE_string('eval', "./resources/gt/1_pred.txt", 'txt file path for evaluation')


def main(_argv):
    # set present path
    home = os.getcwd()
    
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    #initialize deep sort
    # model_filename = 'weights/mars-small128.pb'
    model_filename = os.path.join(home, "weights", "arcface_weights.h5")
    encoder = gdet.create_box_encoder(model_filename, batch_size=128)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []


    # Database 생성
    face_db = dict()

    db_path = FLAGS.database
    for name in os.listdir(db_path):
        name_path = os.path.join(db_path, name)
        name_db = []
        for i in os.listdir(name_path):
            if i.split(".")[1] != "jpg": continue
            id_path = os.path.join(name_path, i)
            img = cv2.imread(id_path)
            # img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_in = tf.expand_dims(img_in, 0)
            # img_in = transform_images(img_in, FLAGS.size)
            # boxes, scores, classes, nums = yolo.predict(img_in)
            boxes = np.asarray([[0, 0, img.shape[0], img.shape[1]]])
            scores = np.asarray([[1]])
            converted_boxes = convert_boxes(img, boxes, scores)
            features = encoder(img, converted_boxes)

            if features.shape[0] == 0: continue

            for f in range(features.shape[0]):
                name_db.append(features[f,:])
        name_db = np.asarray(name_db)
        face_db[name] = dict({"used": False, "db": name_db})
    

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))    
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    count = 0 

    detection_list = []

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)

        # print(boxes, scores, classes, nums)
        # time.sleep(5)
        t2 = time.time()
        times.append(t2-t1)
        print(f'yolo predict time : {t2-t1}')
        times = times[-20:]

        t3 = time.time()
        #############
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0], scores[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

        t4 = time.time()
        print(f'feature generation time : {t4-t3}')

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        t5 = time.time()
        # Call the tracker
        tracker.predict()
        # tracker.update(detections)
        tracker.update(detections, face_db, FLAGS.max_face_threshold)
        t6 = time.time()
        print(f'tracking time : {t6-t5}') 
        
        frame_index = frame_index + 1
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            face_name = track.get_face_name()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id))+len(str(face_name)))*23, int(bbox[1])), color, -1)
            # cv2.putText(img, class_name + face_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.putText(img, class_name + "-" + str(track.track_id) + "-" + face_name, (int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            # cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            # print(class_name + "-" + str(track.track_id))

            # detection_list.append(dict({"frame_no": str(frame_index), "id": str(track.track_id), "x": str(int(bbox[0])), "y": str(int(bbox[1])), "width": str(int(bbox[2])-int(bbox[0])), "height": str(int(bbox[3])-int(bbox[1]))}))
            if face_name != "":
                detection_list.append(dict({"frame_no": str(frame_index), "id": str(face_name), "x": str(int(bbox[0])), "y": str(int(bbox[1])), "width": str(int(bbox[2])-int(bbox[0])), "height": str(int(bbox[3])-int(bbox[1]))}))
        #######
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        # img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
        #                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 20, 255), 2)
        if FLAGS.output:
            out.write(img)
            # frame_index = frame_index + 1
            # list_file.write(str(frame_index)+' ')
            # if len(converted_boxes) != 0:
            #     for i in range(0,len(converted_boxes)):
            #         list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            # list_file.write('\n')
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    
    frame_list = sorted(detection_list, key= lambda x: (int(x["frame_no"]), int(x["id"])))
    # pprint.pprint(frame_list)

    f = open(FLAGS.eval, "w")
    for a in frame_list:
        f.write(a["frame_no"] + " " + a["id"] + " " + a["x"] + " " + a["y"] + " " + a["width"] + " " + a["height"] + "\n")
    # 파일 닫기
    f.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
