## TinyML: Tensorflow lite for microcontroller - Mobility


### 프로젝트명



- Watching U
  - Helmet Detection for mobility   

### 프로젝트 개요   



- 이 프로젝트는 전동킥보드 탑승자의 헬멧 착용 여부를 탐지하고, 착용하지 않은 경우 경고 알람을 내보내는 디바이스를 만드는 프로젝트입니다.



- 헬멧 미착용 상태로 전동 킥보드를 탑승하는 것이 안전상의 이슈로 커져가고 있는 현 상황에서,  
  각각의 개별적인 킥보드를 단속하기 힘들다는 문제를 해결하기 위해 고안되었습니다.



- 디바이스 카메라에 헬멧 미착용 여부가 탐지되면, "헬멧을 쓰세요" 라는 음성이 출력되고,  
  헬멧을 착용한 것으로 판단되면, "헬멧을 잘 쓰셨군요"라는 음성이 출력됩니다.


### 개발 환경 및 부품   



- 개발 환경

  - OS : Raspbian Linux

  - python 3.7 higher 버전이 필요

- 사용 부품
  - Raspberry Pi Model : Raspberry Pi 4 B 1GB RAM
  - Memory : 32GB sd card
  - Camera : RPI NOIR CAMERA BOARD
  - Speaker : LeadSound F10 portable speaker
  - (Option) Accelerator : coral USB accelerator



### 진행 과정

- Tensorflow-lite Model

  - (Step1) Helmet data processing [kaggle-data-processing.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/helmet-data-preprocessing/kaggle-data-processing.ipynb)

    - 데이터  
      - [헬멧 착용 데이터 파일](https://drive.google.com/file/d/1QaMy1wigb7E0T4wPNElUERWwuRg0S186/view)
      - [헬멧 미착용 데이터 파일](https://drive.google.com/file/d/1p7svGkjQfg-p0cIjdMa59KyiYEv7jZVC/view)

    - 수행과정
      - 자, 여기 다양한 경로로부터 수집한 데이터셋이 주어졌습니다.
      - 하지만 우리가 구할 수 있는 최선의 데이터 일부는 우리가 원하는 형식이 아닐 때가 종종 있습니다.
      - 더 복잡한 문제를 해결하기 위해서는, 정말 복잡한 전처리 과정들과 기법이 들어가지만, 데이터를 "사용할 수 있는 형태로 만드는" 전처리를 한번 수행해 보도록 하는 시간입니다.
      - 우리의 요구사항에 맞게 데이터를 변형해 줍니다. (Object Detection → Image Classification)

    - 더 자세한 설명을 원하시면 [이 곳](https://github.com/yunho0130/tensorflow-lite/tree/master/mobilityteamproject/Step1_helmet_data_processing)을 클릭해주세요

  - (Step2) Model training and tflite convert
    - 실습 코드    
      - [helmet_classification_for_tinyMLproject_part1.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/modeling-with-code/helmet_classification_for_tinyMLproject_part1.ipynb)
      - [helmet_classification_for_tinyMLproject_part2.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/modeling-with-code/helmet_classification_for_tinyMLproject_part2.ipynb)
      - [helmet_classification_for_tinyMLproject_part3.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/modeling-with-code/helmet_classification_for_tinyMLproject_part3.ipynb)
    - 수행과정
      - 라즈베리파이와 같이 저사양에서도 충분히 구동할 수 있는 Mobilenet v2 를 학습합니다.
      - 우선 Keras 에서 기본적으로 제공하는 Mobilenet v2 를 활용해서 모델을 만들어 봅니다.
      - 기본으로 제공하는 모델을 개선하기 위해, 직접 Mobilenet v2 구조를 구현하고 모델을 제작해 봅니다.
      - 만들어진 모델을 학습하고, 모델을 텐서플로우 라이트 파일로 변환하는 시간을 갖습니다.
      - (experimental) Grad-Cam 이라는 방법을 통해 해석해 봅니다.


    - 더 자세한 설명을 원하시면 [이 곳](https://github.com/yunho0130/tensorflow-lite/tree/master/mobilityteamproject/Step2_Model_training_and_tflite_convert)을 클릭해주세요

   
- Application
  - (Step3) Raspberry Pi Porting [README 링크](https://github.com/yunho0130/tensorflow-lite/tree/master/mobilityteamproject/Step3_Raspberry_Pi_Porting)  

    - 개발환경 설정
      - 필요 패키지 리스트
        - edgetpu, imutils, numpy, opencv-contrib-python, picamera, Pillow, simpleaudio, tflite-runtime
        - openCV 설치

    - 헬멧 탐지 및 음성 출력 코드 작성
      1. RasberryPi, Coral, Opencv를 설치한다.
      2. Rasbberry Pi 내부에 command창을 켜, ~/opencv4/samples/python 디렉토리로 이동해준다.
      3. 가상환경을 만든다.
      4. 가상환경을 만든 뒤, 위에 나와있는 package들을 설치해준다.
      5. 패키치 설치가 완료되면 이 helmet classification repository git clone한다.
      6. python helmetclassification.py를 실행시킨다.

    - (Option) Google Coral accelerator 연결 및 설치
      - 구글에서 만든 딥러닝 연산 보조장치 (TPU / Tensor Processing Unit) 로써, 라즈베리파이에 연결하여 사용할 수 있다.
      - 사용법
          ```bash
          echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

          curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

          sudo apt-get update

          // USB 가 꽂혀 있으면 우선 다시 뽑아야 함
          // 이미 꽂힌 채로 아래 커맨드를 수행할 시 제거 후 재설치
          // 아래 커맨드는 둘 중 하나 선택.

          sudo apt-get install libedgetpu1-std
          sudo apt-get install libedgetpu1-max
        ```

    - 더 자세한 설명을 원하시면 [이 곳](https://github.com/yunho0130/tensorflow-lite/tree/master/mobilityteamproject/Step3_Raspberry_Pi_Porting)을 클릭해주세요

### reference
  - 헬멧 착용 데이터
    - [캐글 사이트](https://www.kaggle.com/abhishek4273/helmet-dataset)

  - 헬멧 미착용 데이터
    - [Labeled Faces in the Wild Home](http://vis-www.cs.umass.edu/lfw/)
    - [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    - [INRIA Pedestrian](https://dbcollection.readthedocs.io/en/latest/datasets/inria_ped.html)
