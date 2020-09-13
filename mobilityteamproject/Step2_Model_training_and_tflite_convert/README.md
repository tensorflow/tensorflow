# Step2 : 모델 작성, 학습, 평가, 개선, 저장

[2-1-helmet_classification_for_tinyMLproject_part1.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/modeling-with-code/helmet_classification_for_tinyMLproject_part1.ipynb)

[2-2-helmet_classification_for_tinyMLproject_part2.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/modeling-with-code/helmet_classification_for_tinyMLproject_part2.ipynb)

[2-3-helmet_classification_for_tinyMLproject_part3.ipynb](https://colab.research.google.com/github/yunho0130/tensorflow-lite/blob/master/mobilityteamproject/modeling-with-code/helmet_classification_for_tinyMLproject_part3.ipynb)

### 2-1-helmet_classification_for_tinyMLproject_part1.ipynb

- 2-1 을 진행하기 위해 필요한 것들
    - 헬멧 착용 데이터2 : [링크](https://drive.google.com/file/d/1PeyTi_bW23ZSYvybofnON-qOmi5aG0Bi/view?usp=sharing)
    - 헬멧 미착용 데이터2 : [링크](https://drive.google.com/file/d/1p7svGkjQfg-p0cIjdMa59KyiYEv7jZVC/view)
- 2-1 을 마치면 나오는 결과물
    - 작업 경로 파일 : <파일명>
    - 케라스 모델 + 모델 가중치 파일 : helmet_classification_model.h5
- 여기서 무엇을 하나요?
    - Google Drive 와 Google Colab 을 연동시켜 데이터를 불러옵니다.
    - 데이터의 개수를 조정하고, 우리가 만들 모델에 이미지를 넣을 준비를 합니다.
    - 준비를 할 때, 데이터를 요리조리 변형시키면서 다양한 데이터를 입력시키기 위해 노력합니다.
    - tf.keras 에 내장되어 있는 mobilenet v2 모델을 가져오고, 정의해둔 설정을 통해 학습을 수행합니다.
    - Callback : 한 번 training 을 시작하면 에폭수, 데이터의 양, 모델의 크기, 하드웨어의 사양에 따라 짧게는 10분 길게는 몇 달동안 학습이 지속되곤 합니다. 그동안 사람이 모델과 상호작용하는 방법에 대해서 간단히 tensorboard  라는 것을 소개합니다. Callback 이란, 특정 시점이 될 때 실행하도록 특정 시점마다 "등록" 해 두는것을 의미합니다. Tensorboard Callback 을 등록하는 소스코드가 포함되어 있습니다.
- 이 코드에서 어려운 점이 무엇인가요?
    - keras 에 대한 기본적인 개념을 잘 알고 있지 못하다면 어려울 수 있습니다.
    - 딥러닝에 익숙하지 않은 분들은, 많은 하이퍼파라미터들의 종류들에 익숙하지 않을 수 있습니다.
- 결과물을 만들어 내는 과정은 어떻게 구성되어 있나요?
    - (a) keras 모델 파일은 h5 형식으로 쉽게 저장이 가능합니다. 이때, 모델에 학습된 가중치를 모두 담아 함께 저장할 수 있습니다. 우리가 만들고 학습시킨 모델이 포함되어 있습니다.
    - (b) classification 을 포함한 다양한 문제들을 풀 때, 어떤 결과가 무엇인지 이해하는 데 필요한 파일 (예를 들어, 0번 클래스가 많이 활성화된 것은 헬멧을 착용한 클래스가 많이 활성화되었다는 것이고, 1번 클래스가 많이 활성화된 것은 헬멧을 착용하지 않은 클래스가 많이 활성화되었다는 뜻) 이 포함되어 있습니다.
    - (c) 헬멧 착용 데이터 파일2 (Step1-1 결과물) 에서 해상도가 비교적 크기가 큰 이미지 crop 들만 남아 있는 상태로 바뀐 helmet 데이터셋이 포함되어 있습니다. 이때, train 데이터와 test 데이터도 분리되어 있는 상태입니다.
    - (d) 헬멧 미착용 데이터 파일2 (Step1-1 결과물) 이 포함되어 있지만, Step1-1 의 결과물의 모든 하위 디렉터리의 파일들이 전부 train 데이터와 test 데이터로 분리되어 있는 상태입니다.
    - (a) ~ (d) 가 모두 포함되어 압축되어 있는 파일입니다.

### 2-2-helmet_classification_for_tinyMLproject_part2.ipynb

- 2-2 을 진행하기 위해 필요한 것들
    - 작업 경로 파일 : <파일명>
- 2-2 을 마치면 나오는 결과물
    - 케라스 모델 + 모델 가중치 파일 : helmet_classification_model_handmaded_mobilenet.h5
- 여기서 무엇을 하나요?
    - 2-1 에서 tf.keras.applications.mobilenet 을 사용하였다면, 여기서는 mobilenet 구조를 직접 설계합니다.
    - 설계함으로써 얻을 수 있는 이점은, 우리가 풀려고 하는 문제에 더욱 적합한 크기의 모델을 설계할 수 있다는 점입니다.
    - 기존의 mobilnet v2 코드는, 수백 개의 class 분류가 가능할 만큼 복잡한 문제를 풀도록 설계되었습니다. 반면, 우리는 그렇게 복잡한 문제를 푸는 것이 아니고, 헬멧을 썼는지 쓰지 않았는지만 판단하면 되는 문제입니다.
    - 레이어의 깊이를 줄이고, 다양한 parameter 들을 바꾸어 가면서 우리의 문제에 더욱 적절하고 가벼운 모델을 찾을 수 있을 것입니다.
- 이 코드에서 어려운 점이 무엇인가요?
    - 모델을 직접적으로 쌓는 부분이 들어가 있습니다. keras 의 functional API 라는 것을 중심적으로 활용합니다. 오류상황을 방지하기 위한 decorator 문법도 들어가 있지요.
    - 그 외에도, 이미지 데이터 전처리 하는 과정을 일부 추가하는 코드도 사용되었습니다.
    - 가우시안 노이즈를 사용해서 input image 를 강제로 흐릿하게 만드는 코드도 들어가 있습니다. 해상도 차이에 의한 잘못된 학습을 막기 위하여 추가적인 normalization 레이어를 사용합니다.
    - 차근차근 진행하시는 분들께, 어려운 내용일 것이라고 예상합니다. 그렇다면, 모델을 이렇게 쌓는구나! 정도만 이해하고 넘어가 주세요. 다시 공부할 필요가 있을 때 유용하게 사용될 수 있는 코드일 것입니다.
- 결과물을 만들어 내는 과정은 어떻게 구성되어 있나요?
    - 2-1 과 동일합니다.

### 2-3-helmet_classification_for_tinyMLproject_part1.ipynb

- 2-3 을 진행하기 위해 필요한 것들
    - 작업 경로 파일 : <파일명>
    - 케라스 모델 + 모델 가중치 파일 : helmet_classification_model_handmaded_mobilenet.h5
        - helmet_classification_model.h5 를 사용해도 좋으나 일부 소스코드를 수정해야 합니다.
- 2-3 을 마치면 나오는 결과물
    - tflite 모델 파일 : helmet_classification_model_handmaded_mobilenet.tflite
        - helmet_classification_model.h5 을 사용한 경우에는, helmet_classification_model.tflite 가 결과물로 출력됩니다.
- 여기서 무엇을 하나요?
    - .h5 파일에는 keras 로 제작한 모델의 구조와 모델의 가중치가 함께 저장되어 있습니다.
    - 이를 다시 python 으로 불러온 뒤, tensorflow lite 파일로 변환할 수 있는 converter API 를 사용합니다.
    - 이 코드에서는 converter API 를 사용하는 간단한 예를 보여 줍니다. input 과 output 의 자료형을 결정하면서, 어떤 자료형으로 모델과 가중치를 양자화할 것인지 결정합니다. 또한, 양자화 과정에 있어, 우리가 가지고 있는 dataset 의 일부를 제공함으로써 정확한 양자화가 가능하도록 합니다.
    - Step3 에서 google CORAL 이라는 기기에 우리의 모델을 올리기 위해, int8 로 양자화하기 위한 코드가 포함되어 있습니다.
    - grad-cam 이라는 방법을 이용해서, 우리의 모델이 올바른 방법으로 헬멧 착용여부를 추론하고 있는지 검사합니다.
- 이 코드에서 어려운 점이 무엇인가요?
    - grad-cam 이라는 방법에 대해서 익숙하지 않은 분들이 있을 것입니다. 초심자에게 간단한 내용은 아니기 때문에, "특정 클래스로 판단하기 위해 어떤 부분을 주력해서 관찰했는가" 에 대한 정보를 모델로부터 캐내는 방법론 정도로 이해하면 좋을 것 같습니다.
