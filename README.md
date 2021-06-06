
# Introduction

Yolov3와 Arcface 를 통한 얼굴 탐지 및 feature extration을 통해 원하는 사진을 기반으로 영상에서 인물을 찾아 추적하는 모델을 제작하였습니다.

![image](https://user-images.githubusercontent.com/37871541/120924151-373a4500-c70d-11eb-903f-5b2baff60cfb.png)


# 동작 영상

![2 (online-video-cutter com) (1)](https://user-images.githubusercontent.com/37871541/119682508-e6f5f400-be7d-11eb-9623-dadf082302e7.gif)
![up_brunoMars](https://user-images.githubusercontent.com/37871541/120925846-753b6700-c715-11eb-8dae-fc7aa3d3aa02.gif)
![up_Apink](https://user-images.githubusercontent.com/37871541/120925855-7cfb0b80-c715-11eb-8dd8-9e362f638502.gif)
![up_Westlife](https://user-images.githubusercontent.com/37871541/120925858-81272900-c715-11eb-90b2-bbdc9c511e24.gif)
![up_HelloBubble](https://user-images.githubusercontent.com/37871541/120925881-9a2fda00-c715-11eb-8531-d62ca32247e9.gif)
![up_GirlsAloud](https://user-images.githubusercontent.com/37871541/120925884-9c923400-c715-11eb-94cc-1f218ddb2607.gif)
![up_Darling](https://user-images.githubusercontent.com/37871541/120925887-9f8d2480-c715-11eb-97aa-7af27199651a.gif)
![up_T-ara](https://user-images.githubusercontent.com/37871541/120925933-c51a2e00-c715-11eb-8219-89d0585b303a.gif)


# 프로젝트 시작 배경

1.	사람의 얼굴을 지속적으로 추적하는 것은 미래 사회에서 상당히 중요한 기술로 자리매김할 것입니다. 상점에 있는 전광판, 키오스크 등에 장치할 경우, 디스플레이를 쳐다보고 있는 사람 수, 보는 시간 등을 수치적으로 산출할 수 있습니다. 이러한 방법은 광고주가 광고 캠페인을 최적화하는데 사용할 수 있습니다. 
2.	또한 실내 환경의 상태를 추적하는데도 사용이 가능합니다. 공장/사무실과 같은 공간에서 현재 근무하고 있는 사람의 수를 측정하고, 각각의 사람을 인지할 수 있다면, 출근 명부와 같은 단순한 작업을 자동화할 수 있습니다. 또한 자동차와 같은 실내 공간에서 탑승객의 상태를 인지하는데 사용할 수 있습니다. 특히 운전자가 어떤 상태로 주행을 하고 있는지 알 수 있다면, 이는 주행의 안전을 보조하는 하나의 장치로 사용이 가능할 것으로 보입니다.

# 주요 내용
1. Yolov3 구조를 기반으로WIDERFACE dataset을 사용하여 학습한다.
2. 제작된 네트워크를 기반으로 실시간 Detection알고리즘을 제작한다.
3. Deepsort 구조를 개선하여, 얼굴 추적에 맞도록 Feature Descriptor를 Arcface로 대체하여 얼굴에 맞는 feature로 추적 알고리즘을 제작한다. 이 때 임의의 사람에 대해서 고유한 id를 주어 추적하도록 한다.
4. 정적인 사람 이미지에 대해서 Feature 정보를 추출하고, 이를 기반으로 영상에 존재하는 사람을 추적하는 알고리즘을 개발한다. 알고있는 사람의 얼굴 정보가 주어진다면, 해당 사람을 계속해서 추적하도록 한다.

# 변경 사항
1. 실시간 구조 설계
    Deep sort 알고리즘은 탐지된 물체에 대해 추적만을 진행했기 때문에, 이를 실시간 화면을 기반으로 적용하기 위해서는 추가적인 조치가 필요했다. Opencv와 tensorflow를 기반으로 기존의 Deep sort Repository를 fork하여 실시간으로 영상에 대해 처리가 가능하도록 구조를 변경하였다.
2. Yolov3 사용 및 Feature Descriptor 변경
    * Yolov3 사용 : 문제를 종합해본 결과, Detector의 성능이 떨어지는 것이 가장 중요한 문제라고 판단하였다. 이에 Yolov3를 사용하여 얼굴에 국한된 훈련 모델을 제작하였다. 학습 데이터로는 WIDER FACE(A Face Detection Benchmark)를 사용하였다. WIDER FACE 데이터 셋은 얼굴 검출 벤치마크 데이터 셋으로서, 총 32,203개의 이미지를 가지고 있으며, 약 40만개의 얼굴 label로 이루어져 있다. 또한 이 label은 다양한 스케일을 반영하고 있기 때문에, 영상에서 발생하는 다양한 Scale을 반영하기 충분하다고 판단했다.
    * Arcface를 사용한 Feature Descriptor 변경 : 기존 Deep sort는 보행자를 기반으로 해서, Track matching을 진행했기 때문에, FeatureDescriptor의 Input shape이 2:1 비율로 제작되어 있었다. 하지만 실제로 Face Detection의 결과는 사람의 얼굴이기 때문에 정사각형에 가깝다. 또한 기존의 Feature Descriptor의 경우 보행자 이미지를 기반으로 훈련하여 현재 과업에 맞지 않아 실제 테스트를 진행했을 때, id 매칭에 있어서 실패하는 모습을 보였다. 그렇기 때문에 이 부분을 변경하여 실험을 진행하였다. 
3.  Face Matching 알고리즘 제작
    * FACE DB 제작 : 영상속에 등장하는 사람을 인식할 수 있도록 사전에 라벨링된 사람의 이미지를 바탕으로 Feature를 추출하였다. 이 때, Feature 추출기로 얼굴에 최적화된 Arcface 추출기를 사용하였으며, 해당 데이터는 hash 자료형을 기반으로 메모리에 저장되어 있다.![image](https://user-images.githubusercontent.com/37871541/120926027-2215e400-c716-11eb-929b-3a21f8b92eb7.png)

    

    * FACE Matching Algorithm 제작 : 위에서 제작된 Face DB를 바탕으로 런타임에 지속적으로 탐지된 얼굴에 대해 인식을 시도한다. 기존의 Track 객체에는 이전 프레임동안 추적해온 얼굴에 대한 Feature 정보들이 저장되어 있다. 이를 기반으로 Face DB에 있는 사람 얼굴 정보와 Cosine similarity를 통해 가장 작은 값을 해당 Track 객체의 얼굴이라고 판단한다. 이 때, 특정 Threshold(0.68)보다 큰 값일 경우 잘못 탐지했다고 판단하고, id를 배정하지 않는다. 해당 Threshold는 Arcface의 Cosine similarity threshold 실험값을 반영하였다.
    ![image](https://user-images.githubusercontent.com/37871541/120924132-1a9e0d00-c70d-11eb-8b20-02512f01ef77.png)




# Result

[Tracking Persons-of-Interests via Adaptive Discriminative Features(ECCV 2016](https://sites.google.com/site/shunzhang876/eccv16_facetracking/)에서 제작한 Music Video Dataset을 기반으로 성능 테스트를 진행했다. 제작한 모델은 두 가지 기능이 가능하다. 기존 사람 사진에 대한 정보가 없을 때, 영상을 기반으로 고유한 id를 추출하여 추적을 하는 방법, 그리고 Face DB가 주어졌을 때 이를 매칭한 Face id를 매칭하는 방법이다. 이 두가지 방법을 적용하여 테스트를 진행했다. 

![image](https://user-images.githubusercontent.com/37871541/119105685-89663f80-ba58-11eb-89fe-6c29e5f9d4c6.png)
## 이전 연구 결과
|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|-|-|-|70%|89%|-|-|1152|-|61.1%%|65.7%|0.2|


## Face DB없이 고유 ID를 배정한 경우
|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|6.9%|6.98%|6.97%|80.7%|80.87%|2421|2261|348|589|57.4%|68.7%|0.34|

## Face DB를 활용하여 인물을 배정을 우선시 한 경우
|IDF1|IDP|IDR|Rcll|Prcn|FP|FN|IDs|FM|MOTA|MOTP|FAR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|52.6%|60.1%|51.95%|79.35%|81.84%|2220|2434|201|603|59.01%|67.2%|0.32|

추적의 성능을 나타내는 MOTA와 MOTP가 약 60%의 성능을 달성하는 모습을 보였다. 또한 Face DB에서, 등장하는 인물에 대한 사전 정보를 통해 ID를 추출해본 결과, ID에 관련된 지표(IDF1, IDP, IDR) 역시 좋은 결과를 만들었다. 또한 ids에서 기존 보다 좋은 성능을 가진다. 하지만 MOTA에 있어서 FN, FP가 많아 좋은 성능을 가지지 못했다.


# 결론

주요 특징은 크게 세 가지로 구분된다. 
1. 먼저 임의의 사람에 대해 추적이 가능하다. 영상에 몇명의 사람이 존재하는지 모르더라도 등장하는 인물에 대해 고유한 id를 배정할 수 있다. 
2. 두번째로 가려진 경우에도 적은 ID switching를 가진 상태로 연속적인 추적이 가능하다. 하나의 Track이 얼굴을 탐지한 상태일 때, 이 얼굴이 가려진 경우, 30frame 내에는 해당 Track 객체가 메모리에 존재한 상태로 만들어, 만약 30초 내에 재등장한다면 해당 Track id를 배정함으로써 문제를 어느정도 해결했다. 
3. 마지막으로 이미 사람에 대한 정보를 가지고 있다면, 이 정보를 바탕으로 해당 사람을 찾아낼 수 있다. 만약 인물 사진을 가지고 있을 경우, 이 정보를 바탕으로 Feature를 생산하고, 가장 유사한 사람과 매칭할 수 있다.



# Environment Setting

- python 3.7
- tensorflow 2.4.1
- cuda 11.2
- cudnn 8.0.5
- GPU : RTX 3080
- CPU : AMD Ryzen 7 5800X 8-Core Processor

추가적으로 사용한 라이브러리는 아래 파일을 통해 설치할 수 있습니다.

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
