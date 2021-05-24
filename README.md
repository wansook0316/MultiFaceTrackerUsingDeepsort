
# Introduction



# Environment Setting

- python 3.7
- tensorflow 2.4.1
- cuda 11.2
- cudnn 8.0.5
- GPU : RTX 3080
- CPU : AMD Ryzen 7 5800X 8-Core Processor

```bash
$ pip install requirement.txt
```

## If you want Evaluation

[여기](https://drive.google.com/file/d/1_vRwE25Q5Cg7ViaCDr-C9O6ikRZTB0uh/view?usp=sharing)에서 가중치 파일을 받아 `./` 최상위 경로에 압축을 풀어주세요.

### Download Music Video Dataset

[여기](https://drive.google.com/file/d/1v_ZNWEjQEMH87v87Ape7asNbC_fQd2XX/view?usp=sharing)에서 비디오, ground truth 파일을 받아서 최상위 경로에 압축을 풀어주세요. 해당 데이터셋은  [Tracking Persons-of-Interests via Adaptive Discriminative Features(ECCV 2016)](https://sites.google.com/site/shunzhang876/eccv16_facetracking/)에서 가져왔습니다.


### Convert GT xml file to txt

ground truth 파일을 tracking evaluation을 위해 변환해주는 과정을 거칩니다.

```
python xml2txt.py \
    --gt_path ./resources/gt/T-ara_gt.xml \
    --gt_file_path ./resources/gt/T-ara_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/GirlsAloud_gt.xml \
    --gt_file_path ./resources/gt/GirlsAloud_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/Darling_gt.xml \
    --gt_file_path ./resources/gt/Darling_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/Westlife_gt.xml \
    --gt_file_path ./resources/gt/Westlife_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/BrunoMars_gt.xml \
    --gt_file_path ./resources/gt/BrunoMars_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/HelloBubble_gt.xml \
    --gt_file_path ./resources/gt/HelloBubble_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/Apink_gt.xml \
    --gt_file_path ./resources/gt/Apink_gt.txt
```


### Extract Reference Image

다음으로 해당 데이터셋에서 gt에 맞는 얼굴을 추출하기 위해 아래의 코드를 동작시킵니다.

```
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
```

### Let's Tracking

이제 모든 준비가 완료되었습니다. 비디오에 대해서 tracking을 진행합니다.

```
python object_tracker.py \
    --video ./resources/video/in/T-ara.mov \
    --database ./resources/database/T-ara \
    --output ./resources/video/out/T-ara.mp4 \
    --eval ./resources/gt/T-ara_pred.txt

python object_tracker.py \
    --video ./resources/video/in/BrunoMars.mp4 \
    --database ./resources/database/BrunoMars \
    --output ./resources/video/out/BrunoMars.mp4 \
    --eval ./resources/gt/BrunoMars_pred.txt

python object_tracker.py \
    --video ./resources/video/in/Darling.mp4 \
    --database ./resources/database/Darling \
    --output ./resources/video/out/Darling.mp4 \
    --eval ./resources/gt/Darling_pred.txt

python object_tracker.py \
    --video ./resources/video/in/GirlsAloud.mp4 \
    --database ./resources/database/GirlsAloud \
    --output ./resources/video/out/GirlsAloud.mp4 \
    --eval ./resources/gt/GirlsAloud_pred.txt

python object_tracker.py \
    --video ./resources/video/in/HelloBubble.mp4 \
    --database ./resources/database/HelloBubble \
    --output ./resources/video/out/HelloBubble.mp4 \
    --eval ./resources/gt/HelloBubble_pred.txt

python object_tracker.py \
    --video ./resources/video/in/Westlife.mp4 \
    --database ./resources/database/Westlife \
    --output ./resources/video/out/Westlife.mp4 \
    --eval ./resources/gt/Westlife_pred.txt

python object_tracker.py \
    --video ./resources/video/in/Apink.mp4 \
    --database ./resources/database/Apink \
    --output ./resources/video/out/Apink.mp4 \
    --eval ./resources/gt/Apink_pred.txt
```


### Evaluation

이제 생성된 tracking file을 가지고 평가를 진행합니다.

```
python evaluation.py \
    --gt_file_path ./resources/gt/T-ara_gt.txt \
    --pred_file_path ./resources/gt/T-ara_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/GirlsAloud_gt.txt \
    --pred_file_path ./resources/gt/GirlsAloud_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/Darling_gt.txt \
    --pred_file_path ./resources/gt/Darling_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/Westlife_gt.txt \
    --pred_file_path ./resources/gt/Westlife_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/BrunoMars_gt.txt \
    --pred_file_path ./resources/gt/BrunoMars_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/HelloBubble_gt.txt \
    --pred_file_path ./resources/gt/HelloBubble_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/Apink_gt.txt \
    --pred_file_path ./resources/gt/Apink_pred.txt
```




# Reference

[deepface](https://github.com/serengil/deepface)
[Deep SORT](https://github.com/nwojke/deep_sort)
[YoloV3 Implemented in TensorFlow 2.0](https://github.com/zzh8829/yolov3-tf2)
[YOLOFace](https://github.com/sthanhng/yoloface)
[Tracking Persons-of-Interests via Adaptive Discriminative Features(ECCV 2016)](https://sites.google.com/site/shunzhang876/eccv16_facetracking/)
[Simple Online and Realtime Tracking with a Deep Association Metric(IEEE 2016)](https://arxiv.org/abs/1703.07402)
[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
