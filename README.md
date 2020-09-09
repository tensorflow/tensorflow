# TinyML: Tensorflow lite for microcontroller

[![Yes24](./images/tinyML_bookcover_kor.jpg)](https://www.yes24.com/Product/Goods/91879171)

<!--
[![Oreilly](./images/tinyML_bookcover_eng.jpg)](https://learning.oreilly.com/library/view/tinyml/9781492052036/)
-->

TinyML 번역서의 한국 독자들을 위한 한글 소스코드 저장소를 개설하게 되었습니다. 책에서도 명시하고 있지만, 텐서플로우 프로젝트는 업데이트가 활발히 진행되고 있는 프로젝트입니다. 가장 최신 코드는 아래의 영어 원문 코드를 참조하시기 바랍니다.

- TinyML(Tensorflow Lite) <https://oreil.ly/TQ4CC>

## 실습소스코드

### Google Colab 통한 실습
아래의 Jupyter Notebook 파일은 Google Colab과 연결되도록 만들어두었습니다. 이를 통해 책의 주요 딥러닝 모델 학습 코드를 간편하게 실행해 볼 수 있습니다.

- __Ch04: 사인파 예측하는 모델 만들기__ [create_sine_model_ko.ipynb ](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/tensorflow/lite/micro/examples/hello_world/create_sine_model_ko.ipynb)

    ![](./images/sinewave.png)

- __Ch08: 음성 인식 모델 만들기__ [train_speech_model.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/tensorflow/lite/micro/examples/micro_speech/train_speech_model_ko.ipynb)

    ![](./images/training_audio.gif)
    ![](./images/cross_entropy.gif)

- __Ch12: 마술 지팡이 제스쳐 인식하기__ [train_magic_wand_model.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/tensorflow/lite/micro/examples/magic_wand/train/train_magic_wand_model_kor.ipynb)

    ![](./images/training.gif)
    ![](./images/tensorboard.gif)


### 마이크로컨트롤러 기기를 통한 실습  
기기에서 동작하게 될 소스코드는 본 소스코드 저장소를 직접 다운로드 받은 후 압축을 푼 뒤 아두이노와 같은 마이크로컨트롤러 기기에 업로드 하여 사용할 수 있습니다. 자세한 실습 방법은 책 혹은 아래의 공식개발문서를 참고하세요.

- __소스코드 저장소 다운받기__ [tensorflow-lite-kor-master.zip](https://github.com/yunho0130/tensorflow-lite/archive/master.zip)

    ![](./images/microcontroller.png)

### 마이크로컨트롤러용 텐서플로우 라이트 (TensorFlow Lite for Microcontrollers)

`마이크로 컨트롤러용 텐서플로우 라이트 (TensorFlow Lite for Microcontrollers)`는 아주 작은 메모리(KB)를 사용하는 기기에서 머신러닝모델을 실행하도록 텐서플로우 라이트(TensorFlow Lite)를 이식한 프레임워크입니다.
- 이 프레임워크에 대해 더 자세히 배우고 싶다면 공식개발문서를 확인하세요. [tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers).

### Project Showcase

# Pull Request

Contributor: 강상훈(sanghunkang), 공예슬(0ys), 김도연(yammayamm), 김창윤(Karmantez), 김하림(harheem), 송보영(ufo8945), 서미지(prograsshopper), 이보성(dlqh406), 양윤석(yoonseok312)

### 프로젝트 배경 (프로젝트 선정이유, 키워드 및 선정이유)

Watch-out 은 Tensorflow Lite 모델을 이용해 인터넷에 접속이 안되는 상황에서도 작동할 수 있는 아이폰 & 애플워치 어플리케이션입니다. 인식할 수 있는 키워드에는 2020년 9월 기준 "불이야" 와 "수지"가 있습니다. 

"불이야" - 청각장애인들이 화재상황에서 "불이야!"라는 소리를 듣지 못하여 대처가 늦거나 위험에 빠질 수도 있다는 기사 내용을 바탕으로 키워드를 선정하였습니다. 

"수지" -  주변 사람들이 자신의 이름을 부르는 것도 듣지 못한다는 것 또한 불편사항이 될 수도 있다는 생각을 바탕으로 watch-out이 대신 사용자의 이름을 듣도록 하였습니다. 발음하기 쉽고 다른 이름과 헷갈리지 않는 대표적인 이름으로 '수지'를 선택하였습니다.

### 모델

- **데이터 준비, 데이터 형식**
    - 기존 모델에 들어있던 데이터를 바탕으로 데이터의 형식과 데이터의 개수를 정하였습니다.  데이터 개수는 한 키워드당 2000개를 목표로 하였고, 데이터 형식은 1초 미만의 wav 파일을 준비하도록 하였습니다. 한 사람당 키워드 별로 200개씩 총 400개의 녹음파일을 준비하였고, 모델이 오버피팅이 되지 않도록 최대한 다양한 사람의 목소리, 다양한 어조와 톤, 빠르기로 녹음을 하였습니다.
- **트레이닝 방식 및 커맨드**
    - 훈련

        train.py를 실행시킬 때 플래그값들을 조절해서 모델을 커스텀화 할 수 있습니다. 여기서 중요한 부분들은 다음과 같습니다.

        1. `—model_architecture` 어떤 네트워크를 사용할 것인지 지정하는 플래그. CNN을 기반으로한 여러 변형형태 중 하나를 선택할 수 있다. 본 프로젝트에서는 iOS 예제에서 사용하고 있는 모델인 "conv"모델을 사용하였습니다.
        2. `—wanted_words` bulyiya, suzy
        3. `—data_dir` 훈련에 사용할 데이터가 저장되어 있는 디렉토리. —wanted_words에서 인식하고자 한 단어들은 —data_dir에서 지정한 경로 안에 같은 이름으로 된 디렉토리를 만들고 그 안에 훈련에 사용할 파일들을 저장해야 합니다. 훈련데이터는 16bit-wav의 형식을 따라야 합니다.
        4. `—train_dir` 훈련결과를 저장할 디렉토리. checkpoint 파일이 여기서 지정한 경로에 저장됩니다. 
        5. `—how_many_training_steps` epoch. 데이터를 몇 바퀴 돌려서 학습할 것인지 설정합니다.
    - freeze

        아까 뭐가 생겼다 없어졌는데 ㅋㅋㅋㅋ 착각입니다 ^0^ ㅋㅎㅎㅎㅎㅎ

    - interpreter compile
    - 인퍼런스
        - pb
        - tflite
- **Issue**
    - Tensorflow version issue

        2.x version부터는 tflite_converter에서 saved_model의 변환만을 지원하기 때문에 기존 모델 생성에서 문제가 발생하였습니다.

        speech_commands 예제는 1.x tensorflow에 맞춰서 개발되었습니다. 따라서 speech_commands 예제에서 사용하는 모델 훈련결과 저장방식은 checkpoint입니다. 

        현재(2020.09) tensorflow 설치시 특별한 설정을 하지 않으면 2.x version tensorflow가 설치됩니다. 따라서 교재에서 사용하는 명령어들이 일부 작동하지 않을 수 있습니다.

        - SOLUTION 1: checkpoint를 saved_model형식으로 변환
        - **SOLUTION 2: tensorflow 1.x대로 다운그레이드 (채택)**
    - Data count issue
    - Recognition error & Rank issue

### 어플리케이션

아이폰 앱과 워치 앱을 구동시키면 스플래시 화면 이후 메인 화면이 나타나게 됩니다. 메인 화면에서 토글을 키면 입력되는 소리가 텐서플로우 모델로 들어가 inference 과정을 거칩니다. inference 후 아이폰과 워치에 각각 인식한 단어를 전해주는 알림이 전해지고, 알림은 5초 후에 꺼지거나 사용자가 탭하면 꺼집니다. 위험한 소리의 경우, 사용자가 곧바로 119에 전화할 수도 있습니다. 또한, 설정 페이지에서 인식하고 싶은 단어와 무시하고 싶은 단어를 설정할 수 있습니다. 

- iOS → Watch 간의 데이터 전송
    - iOS: MainViewModel에서 tensorflow 모델을 구동하고, 그 결과값을 WCSession을 이용해 watch에게 넘겨줍니다.
    - Watch: WCSession에서 받은 텍스트를 WatchEnvironment라는 Watch 전체에서 쓸 수 있는 변수로 선언하여 WatchView 화면에 나타날 수 있게 했습니다.
- viewcontroller → viewmodel / swiftUI 리펙터링
    - 기존 tensorflowlite-ios 어플리케이션 및 타 레퍼런스의 경우 대부분 view controller 와 UIKit 을 이용하는 방식으로 되어 있기에, 최신 UI/UX 적용을 위해 SwiftUI 와 MVVM 아키텍쳐로 코드 리펙터링을 진행하였습니다.
- gif 올리기

- Reference


### Collaborators

> yunho0130(맹윤호), harheem(김하림), prograsshopper(서미지), 0ys(공예슬), yoonseok312(양윤석), dlqh406(이보성), Karmantez(김창윤), kyunghwanleethebest(이경환), new-w(최예진), su-minn(전수민), ProtossDragoon(이장후), yammayamm(김도연), ufo8945(송보영), pmcsh04(조승현), sanghunkang(강상훈)


* TinyML: Tensorflow lite for microcontroller   컨트리뷰션은 아래의 `Github 가이드라인`을 따릅니다.
    - Github 공식 오픈소스 컨트리뷰션 가이드라인 <https://opensource.guide/ko/how-to-contribute/>

### 인용 Citation
본 레파지토리나 <초소형 머신러닝 TinyML>의 내용을 인용하실 때에는 아래의 인용정보를 사용하시면 편리합니다.
```
@book{TinyML-Machine-Learning-with-TensorFlow-Lite,
  title={초소형 머신러닝 TinyML: 모델 최적화부터 에지 컴퓨팅까지 작고 빠른 딥러닝을 위한 텐서플로 라이트},
  author={피트 워든, 대니얼 시투나야케， 맹윤호(역), 임지순(역)},
  isbn={9791162243411},
  url={https://www.hanbit.co.kr/media/books/book_view.html?p_code=B3963656224},
  year={2020},
  publisher={한빛미디어}
}
```
