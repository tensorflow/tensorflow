# TFLITE 헬멧감지기

 ##### - by 오픈소스 컨트리뷰톤 Tensorflow Lite for Microcontroller - Mobility팀



## 1. Raspberry Pi에 OS 설치 (Raspbian)

이미지 및 balenaetcher를 개발환경에 맞게 다운로드

- [라즈베리파이 OS 설치 이미지 다운로드](https://www.raspberrypi.org/downloads/)

- [balenaetcher 다운로드](https://www.balena.io/etcher/)

  ```bash
  #1 balenaetcher를 실행하여 라즈베리파이에 넣을 sd카드에 이미지를 씌운다.(sd카드에 flash만 해주면 됨)

  #2 sd카드를 라즈베리파이에 넣은 후 라즈베리파이를 부팅한다.

  #3 라즈비안 내의 터미널을 사용하여 아래의 개발과정을 진행한다.
  ```

## 2. 개발 사양 & Packages

  #### 개발 환경 및 부품

  - OS : Raspbian Linux

  - Raspberry Pi Model : Raspberry Pi 4 B 1GB RAM

  - Memory : 32GB sd card

  - Camera : RPI NOIR CAMERA BOARD

  - Speaker : LeadSound F10 portable speaker

  - Accelerator : coral USB accelerator

  - python 3.7 higher 버전이 필요

  - Packages Install

  #### 설치 Package list

  - edgetpu @ file:///home/pi/Downloads/edgetpu-2.12.1-py3-none-any.whl
  - imutils==0.5.3
  - numpy==1.19.1
  - opencv-contrib-python==4.1.0.25
  - picamera==1.13
  - Pillow==7.2.0
  - simpleaudio==1.0.4
  - tflite-runtime @ https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl   

  #### ※ requirements.txt에 명시된 Package를 설치하는 방법



  ```bash
  pip3 install -r requirements.txt
  ```





  #### ※ 혹시라도 가상환경(Virtual environment) 생성을 모른다면?

  ```bash
  # 1 코드를 실행 할 디렉토리에 들어가서 venv라는 이름의 가상환경을 생성한다.
  python3 -m venv venv

  # 2 가상환경을 실행시켜준다.
  (디렉토리에 들어갔다는 가정) source bin/activate

  # 3 가상환경 실행 후, pip를 설치해준다.
  pip3 -r install requirements.txt

  # ** 가상환경을 끄는 방법도 있다.
  source deactivate

  ```



  ####    Coral

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

  #### 		Opencv 설치법

  ```bash
  //https://webnautes.tistory.com/916 사이트 참고하시면 됩니다!

  1. OpenCV 컴파일 전 필요한 패키지 설치

  // 우선 기존에 패키치 리스트를 업데이트합니다.
  sudo apt-get update
  sudo apt-get upgrade

  // OpenCV 컴파일 전 필요한 패키지 설치
  sudo apt-get install build-essential
  sudo apt-get install cmake

  // 특정 포멧의 이미지 파일을 Read, Write 하기 위한 필요 패키지 설치
  sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev

  // 특정 코덱의 비디오 파일을 Read, Write 하기 위한 필요 패키지 설치
  sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev

  // Video4Linux  리눅스에서 실시간 비디오 캡처 및 비디오 디바이스 제어를 위한 API 패키지 설치
  sudo apt-get install libv4l-dev v4l-utils

  // GStreamer는 리눅스 기반에서 영상 스트리밍을 쉽게 처리할 수 있더록 만든 오픈 소스 프레임워크 이다.
  sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

  // OpenCV에서 윈도우 생성 등의 GUI를 위해 gtk 또는 qt를 선택해서 사용가능하며 여기서는 gtk2를 지정해준다.
  sudo apt-get install libgtk2.0-dev

  // OpenGL 지원하기 위해 필요한 라이브러리 설치
  sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev

  // OpenCV 최적화를 위해 사용되는 라이브러리 설치
  sudo apt-get install libatlas-base-dev gfortran libeigen3-dev

  2. OpenCV 설정과 컴파일 및 설치
  ~/$ mkdir opencv
  ~/$ cd opencv

  // OpenCV 4.1.2 소스코드 다운로드
  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
  unzip opencv.zip

  // opencv_contrib(extra modules) 소스코드를 다운로드 받아 압축을 풀어줍니다.
  $ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
  $ unzip opencv_contrib.zip

  ```



## 3. 실행방법

1. (위에 나와있는 설치법대로) RasberryPi, Coral, Opencv를 설치한다.
2. Rasbberry Pi 내부에 command창을 켜,  ~/opencv4/samples/python 디렉토리로 cd해준다.
3. 그 후 가상환경을 만든다
4. 가상환경을 만든 뒤, 위에 나와있는 package들을 설치해준다
5. 패키치 설치가 완료되면 이 [helmet classification repository](https://github.com/tinyml-mobility/helmetclassifcation)  git clone한다.
6. python helmetclassification.py를 실행시킨다! (실행 중지시키고 싶으면 Ctrl+C를 누른다.)



## 4. Reference

- face detection 코드 참고: [facedetection](https://github.com/opencv/opencv/tree/master/samples/python) -> facedetect.py
- image classification 코드 참고: [image classification](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)
- OpenCV설치: https://webnautes.tistory.com/916
