# Modeling with code
환경 : Google Colab

2-1-helmet_classification_for_tinyMLproject_part1.ipynb

2-2-helmet_classification_for_tinyMLproject_part1.ipynb

2-3-helmet_classification_for_tinyMLproject_part1.ipynb

### 2-1-helmet_classification_for_tinyMLproject_part1.ipynb

- 2-1 을 진행하기 위해 필요한 것들
    - 헬멧 착용 데이터2 : <파일명>
    - 헬멧 미착용 데이터2 : <파일명>
- 2-1 을 마치면 나오는 결과물
    - 작업 경로 파일 : <파일명>
- 여기서 무엇을 하나요?
    - Google Drive 와 Google Colab 을 연동시켜 데이터를 불러옵니다.
    - 데이터의 개수를 조정하고, 우리가 만들 모델에 이미지를 넣을 준비를 합니다.
    - 준비를 할 때, 데이터를 요리조리 변형시키면서 다양한 데이터를 입력시키기 위해 노력합니다.
    - tf.keras 에 내장되어 있는 mobilenet v2 모델을 가져오고, 정의해둔 설정을 통해 학습을 수행합니다.
    - Callback : 한 번 training 을 시작하면 에폭수, 데이터의 양, 모델의 크기, 하드웨어의 사양에 따라 짧게는 10분 길게는 몇 달동안 학습이 지속되곤 합니다. 그동안 사람이 모델과 상호작용하는 방법에 대해서 간단히 tensorboard  라는 것을 소개합니다. Callback 이란, 특정 시점이 될 때 실행하도록 특정 시점마다 "등록" 해 두는것을 의미합니다. Tensorboard Callback 을 등록하는 소스코드가 포함되어 있습니다.
- 어려운 점이 무엇인가요?
    - keras 에 대한 기본적인 개념을 잘 알고 있지 못하다면 어려울 수 있습니다.
    - 가장 먼저 머신러닝 모델을 처음 접한 사람은
- 결과물은 어떻게 생성되나요?

### 2-2-helmet_classification_for_tinyMLproject_part2.ipynb

- 2-2 을 진행하기 위해 필요한 것들
- 2-2 을 마치면 나오는 결과물
- 어려운 점이 무엇인가요?
- 결과물은 어떻게 생성되나요?

### 2-3-helmet_classification_for_tinyMLproject_part1.ipynb

- 2-3 을 진행하기 위해 필요한 것들
- 2-3 을 마치면 나오는 결과물
- 어려운 점이 무엇인가요?
- 결과물은 어떻게 생성되나요?

- tensorflow 모델 만들기
    - 연관 파일 1-2-
    - 연관 파일 1-3-
    - 연관 파일
    - teachable machine
        - [teachable machine 페이지 링크](https://teachablemachine.withgoogle.com/train)
        - [결과파일 링크](https://drive.google.com/drive/u/0/folders/1HNxIY8bJfM3V29yvG2_W32y51BIH3PQJ)
    - 직접 keras 로 구현
        - [결과물 코드 Github 링크](https://github.com/tinyml-mobility/modeling-with-code/blob/master/helmet_classification_for_tinyMLproject_part2.ipynb)
    - 위 모델을 tflite로 변환
        - 변환하는 코드
