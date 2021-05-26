
# Introduction

Yolov3와 Arcface 를 통한 얼굴 탐지 및 feature extration을 통해 원하는 사진을 기반으로 영상에서 인물을 찾아 추적하는 모델을 제작하였습니다.

## 동작 영상
<iframe width="560" height="315" src="https://www.youtube.com/embed/cCJJJ5JTQx0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 프로젝트 시작 배경

1.	사람의 얼굴을 지속적으로 추적하는 것은 미래 사회에서 상당히 중요한 기술로 자리매김할 것이다. 상점에 있는 전광판, 키오스크 등에 장치할 경우, 디스플레이를 쳐다보고 있는 사람 수, 보는 시간 등을 수치적으로 산출할 수 있다. 이러한 방법은 광고주가 광고 캠페인을 최적화하는데 사용할 수 있다. 
2.	또한 실내 환경의 상태를 추적하는데도 사용이 가능하다. 공장/사무실과 같은 공간에서 현재 근무하고 있는 사람의 수를 측정하고, 각각의 사람을 인지할 수 있다면, 출근 명부와 같은 단순한 작업을 자동화할 수 있다. 또한 자동차와 같은 실내 공간에서 탑승객의 상태를 인지하는데 사용할 수 있다. 특히 운전자가 어떤 상태로 주행을 하고 있는지 알 수 있다면, 이는 주행의 안전을 보조하는 하나의 장치로 사용이 가능하다.

## Architecture

![image](https://user-images.githubusercontent.com/37871541/119675545-19045780-be78-11eb-9591-ec118a6fb493.png)

![image](https://user-images.githubusercontent.com/37871541/119678550-94670880-be7a-11eb-9c49-7add81dc4a92.png)


변경 사항은 다음과 같다.

1. Deep sort의 pedestrian appearance extractor를 Arcface를 통한 feature extraction을 통해 얼굴에 최적화된 모델로 변경한다.
2. 실시간 tracking을 위해 얼굴 탐지 모델을 WIDER FACE 데이터셋을 통해 YOLOv3모델을 학습한다.
3. 학습된 모델을 통해 얼굴을 탐지하고 기존의 deepsort 를 변경하여 실시간 tracking이 가능하도록 한다.
4. 저장된 사람의 얼굴을 feature extraction하여 face db에 저장한다.
5. track객체가 생성될 때 해당 db를 근간으로 가장 높은 face id를 매칭한다. 이 때 cosine similarity를 사용한다.
6. 이미 track이 된 객체에 대해서도 3의 방법을 통해 지속적인 id 갱신을 시도한다.
7. track객체에 이전에 탐색된 얼굴 feature정보를 저장하여 인식률을 높힌다.
8. 이미 face db에서 매칭된 사람의 경우 id가 중복되지 않도록 한다.

## Result

Recall과 precision같은 경우 기존 Tracking 연구보다 높은 수준을 보였으나, MOTA (추적 정확도) 측면에서 나쁜 점수가 나왔다. 이는 각각의 프레임에서 추가적인 얼굴 탐지 때문에 잘못된 사람에 대해 id 매칭이 되어 FN가 크게 오르는 결과를 만들었다. 아래는 [기존 논문](https://sites.google.com/site/shunzhang876/eccv16_facetracking/)의 결과이다.

![image](https://user-images.githubusercontent.com/37871541/119105685-89663f80-ba58-11eb-89fe-6c29e5f9d4c6.png)


해당 논문에서 사용한 music dataset에 대해 제작한 모델의 성능을 검증하였다.

### T-ara

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|53.6%|59.0%|49.5%|76.6%|90.4%|1176|3406|3752|517|42.6%|71%|0.241|4710|

### GirlsAloud

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|39.0%|42.6%|36.4%|73.9%|85.3%|2087|4275|4687|1122|32.6%|64.6%|0.314|6630|

### Darling

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|3.2%|44.2%|42.6%|79.7%|82.1%|1654|1935|3048|743|30.4%|65.7%|0.267|6180|

### Westlife

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|64.3%|61.3%|68.4%|87.8%|77.9%|2828|1389|1809|562|47.0%|64.7%|0.411|6870|

### BrunoMars

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|40.5%|40.7%|40.8%|74.1%|73.1%|4560|4330|5128|1010|16.1%|78.9%|0.539|8460|

### HelloBubble

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|41.9%|45.3%|39.1%|73.9%|85.2%| 673|1363|1381|301|34.6%|69.7%|0.256|4920|

### Apink

|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|Fn|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|56.2% | 58.9% | 53.8% |79.5% |86.8% |  883| 1491| 1234 | 337 |50.4% | 66.8% | 0.15 |4650|



# Environment Setting

- python 3.7
- tensorflow 2.4.1
- cuda 11.2
- cudnn 8.0.5
- GPU : RTX 3080
- CPU : AMD Ryzen 7 5800X 8-Core Processor

추가적으로 사용한 라이브러리는 아래 파일을 통해 설치할 수 있다.

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

다음으로 해당 데이터셋에서 gt에 맞는 얼굴을 추출하여 face db에 반영하기 위해 아래의 코드를 동작시킵니다.

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
