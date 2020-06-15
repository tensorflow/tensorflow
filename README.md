# TinyML: Tensorflow lite for microcontroller 

![https://learning.oreilly.com/library/view/tinyml/9781492052036/](./images/tinyML_bookcover_eng.jpg)

TinyML 번역서의 한국 독자들을 위한 한글 소스코드 저장소를 개설하게 되었습니다. 책에서도 명시하고 있지만, 텐서플로우 프로젝트는 업데이트가 활발히 진행되고 있는 프로젝트입니다. 가장 최신 코드는 아래의 영어 원문 코드를 참조하시기 바랍니다. 

- TinyML(Tensorflow Lite) <https://oreil.ly/TQ4CC>

## 실습소스코드 

### Google Colab 통한 실습 
아래의 Jupyter Notebook 파일은 Google Colab과 연결되도록 만들어두었습니다. 이를 통해 책의 주요 딥러닝 모델 학습 코드를 간편하게 실행해 볼 수 있습니다. 

- __Ch04: 사인파 예측하는 모델 만들기__ [create_sine_model_ko.ipynb ](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/tensorflow/lite/micro/examples/hello_world/create_sine_model_ko.ipynb)

    ![](./images/sinewave.png)

- __Ch08: 음성 인식 모델 만들기__ [train_speech_model.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/tensorflow/lite/micro/examples/micro_speech/train_speech_model.ipynb)

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

### Contributors 

```
yunho0130 
```

* TinyML: Tensorflow lite for microcontroller   컨트리뷰션은 아래의 `Github 가이드라인`을 따릅니다. 
    - Github 공식 오픈소스 컨트리뷰션 가이드라인 <https://opensource.guide/ko/how-to-contribute/>


