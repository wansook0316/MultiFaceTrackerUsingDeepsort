import cv2
import os
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS



flags.DEFINE_string('gt_file_path', './resources/gt/T-ara_gt.txt', 'path to crop gt file')
flags.DEFINE_string('video_file_path', './resources/video/in/T-ara.mov', 'path to video file')
flags.DEFINE_string('face_data_path', './resources/database/T-ara', 'path to video file')

"""
python generate_face.py \
    --gt_file_path ./resources/gt/T-ara_gt.txt \
    --video_file_path ./resources/video/in/T-ara.mov \
    --face_data_path ./resources/database/T-ara

python generate_face.py \
    --gt_file_path ./resources/gt/GirlsAloud_gt.txt \
    --video_file_path ./resources/video/in/GirlsAloud.mp4 \
    --face_data_path ./resources/database/GirlsAloud

python generate_face.py \
    --gt_file_path ./resources/gt/Darling_gt.txt \
    --video_file_path ./resources/video/in/Darling.mp4 \
    --face_data_path ./resources/database/Darling

python generate_face.py \
    --gt_file_path ./resources/gt/Westlife_gt.txt \
    --video_file_path ./resources/video/in/Westlife.mp4 \
    --face_data_path ./resources/database/Westlife

python generate_face.py \
    --gt_file_path ./resources/gt/BrunoMars_gt.txt \
    --video_file_path ./resources/video/in/BrunoMars.mp4 \
    --face_data_path ./resources/database/BrunoMars

python generate_face.py \
    --gt_file_path ./resources/gt/HelloBubble_gt.txt \
    --video_file_path ./resources/video/in/HelloBubble.mp4 \
    --face_data_path ./resources/database/HelloBubble

python generate_face.py \
    --gt_file_path ./resources/gt/Apink_gt.txt \
    --video_file_path ./resources/video/in/Apink.mp4 \
    --face_data_path ./resources/database/Apink

"""
def main(args):
    f = open(FLAGS.gt_file_path, "r")
    detections = []
    while True:
        line = f.readline()
        if not line: break
        a = list(map(int, line.split()))
        detections.append(a)
    detections = np.asarray(detections)
    f.close()

    if not os.path.isdir(FLAGS.face_data_path):
        os.mkdir(FLAGS.face_data_path)

    vid = cv2.VideoCapture(FLAGS.video_file_path)
    frame_index = -1
    count = 0 
    frame_indices = detections[:, 0].astype(np.int)

    object_dict = dict()

    while True:
        frame_index += 1
        print(f'{frame_index} frame is working on...')
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            count+=1
            if count < 3:
                continue
            else: 
                break
        
        mask = frame_indices == frame_index

        

        for row in detections[mask]:
            frame, id, bbox = row[0], row[1], row[2:]

            if object_dict.get(id):
                file_name = object_dict[id]
                object_dict[id] += 1
            else:
                object_dict[id] = 1
                file_name = object_dict[id]

            if object_dict[id] % 10 != 0:
                continue

            # target_aspect = float(img.shape[1]) / img.shape[0]
            # new_width = target_aspect * bbox[3]
            # bbox[0] -= (new_width - bbox[2]) / 2
            # bbox[2] = new_width
            bbox[2:] += bbox[:2]
            bbox = bbox.astype(np.int)

            bbox[:2] = np.maximum(0, bbox[:2])
            bbox[2:] = np.minimum(np.asarray(img.shape[:2][::-1]) - 1, bbox[2:])

            sx, sy, ex, ey = bbox
            # print(bbox)
            # print(img.shape)
            image = img[sy:ey, sx:ex]

            output_path = os.path.join(FLAGS.face_data_path, str(id))
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            cv2.imwrite(os.path.join(FLAGS.face_data_path, str(id), str(object_dict[id])+".jpg"), image)
        


    # frame_indices = detection_mat[:, 0].astype(np.int)
    #     mask = frame_indices == frame_idx

    #     detection_list = []
    #     for row in detection_mat[mask]:
    #         bbox, confidence, feature = row[2:6], row[6], row[10:]
    #         if bbox[3] < min_height:
    #             continue
    #         detection_list.append(Detection(bbox, confidence, feature))
    #     return detection_list
        

 
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


