## Speech Model Tests

Sample test data has been provided for speech related models in Tensorflow Lite
to help users working with speech models to verify and test their models.

### Models and Inputs and Outputs:

[ASR AM model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_asr_am_model.tflite)

[ASR AM quantized model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_asr_am_model_int8.tflite)

[ASR AM test inputs](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_asr_am_model_in.csv)

[ASR AM test outputs](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_asr_am_model_out.csv)

[ASR AM int8 test outputs](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_asr_am_model_int8_out.csv)

The models below are not maintained.

[Speech hotword model (Svdf
rank=1)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_hotword_model_rank1_2017_11_14.tflite)

[Speech hotword model (Svdf
rank=2)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_hotword_model_rank2_2017_11_14.tflite)

[Speaker-id
model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_speakerid_model_2017_11_14.tflite)

[TTS
model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_tts_model_2017_11_14.tflite)

### Test Bench

[Model tests](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/models/speech_test.cc)

Download the ASR AM test models and inputs and output files to the
models/testdata directory to run the tests.


## Speech Model Architectures

For the hotword, speaker-id and automatic speech recognition sample models, the
architecture assumes that the models receive their input from a speech
pre-processing module. The speech pre-processing module receives the audio
signal and produces features for the encoder neural network and uses some
typical signal processing algorithms, like FFT and spectral subtraction, and
ultimately produces a log-mel filterbank (the log of the triangular mel filters
applied to the power spectra). The text-to-speech model assumes that the inputs
are linguistic features describing characteristics of phonemes, syllables,
words, phrases, and sentence. The outputs are acoustic features including
mel-cepstral coefficients, log fundamental frequency, and band aperiodicity.
The pre-processing modules for these models are not provided in the open source
version of TensorFlow Lite.

The following sections describe the architecture of the sample models at a high
level:

### Hotword Model

The hotword model is the neural network model we use for keyphrase/hotword
spotting (i.e. "okgoogle" detection). It is the entry point for voice
interaction (e.g. Google search app on Android devices or Google Home, etc.).
The speech hotword model block diagram is shown in Figure below. It has an input
size of 40 (float), an output size of 7 (float), one Svdf layer, and four fully
connected layers with the corresponding parameters as shown in figure below.

![hotword_model](hotword.svg "Hotword model")

### Speaker-id Model

The speaker-id model is the neural network model we use for speaker
verification. It runs after the hotword triggers. The speech speaker-id model
block diagram is shown in Figure below. It has an input size of 80 (float), an
output size of 64 (float), three Lstm layers, and one fully connected layers
with the corresponding parameters as shown in figure below.

![speakerid_model](speakerid.svg "Speaker-id model")

### Text-to-speech (TTS) Model

The text-to-speech model is the neural network model used to generate speech
from text. The speech text-to-speech model’s block diagram is shown
in Figure below. It has and input size of 334 (float), an output size of 196
(float), two fully connected layers, three Lstm layers, and one recurrent layer
with the corresponding parameters as shown in the figure.

![tts_model](tts.svg "TTS model")

### Automatic Speech Recognizer (ASR) Acoustic Model (AM)

The acoustic model for automatic speech recognition is the neural network model
for matching phonemes to the input audio features. It generates posterior
probabilities of phonemes from speech frontend features (log-mel filterbanks).
It has an input size of 320 (float), an output size of 42 (float), five LSTM
layers and one fully connected layers with a Softmax activation function, with
the corresponding parameters as shown in the figure.

![asr_am_model](asr_am.svg "ASR AM model")

### Automatic Speech Recognizer (ASR) Language Model (LM)

The language model for automatic speech recognition is the neural network model
for predicting the probability of a word given previous words in a sentence.
It generates posterior probabilities of the next word based from a sequence of
words. The words are encoded as indices in a fixed size dictionary.
The model has two inputs both of size one (integer): the current word index and
next word index, an output size of one (float): the log probability. It consists
of three embedding layer, three LSTM layers, followed by a multiplication, a
fully connected layers and an addition.
The corresponding parameters as shown in the figure.

![asr_lm_model](asr_lm.svg "ASR LM model")

### Endpointer Model

The endpointer model is the neural network model for predicting end of speech
in an utterance. More precisely, it generates posterior probabilities of various
events that allow detection of speech start and end events.
It has an input size of 40 (float) which are speech frontend features
(log-mel filterbanks), and an output size of four corresponding to:
speech, intermediate non-speech, initial non-speech, and final non-speech.
The model consists of a convolutional layer, followed by a fully-connected
layer, two LSTM layers, and two additional fully-connected layers.
The corresponding parameters as shown in the figure.
![endpointer_model](endpointer.svg "Endpointer model")
