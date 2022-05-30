# Integrate audio classifiers

Audio classification is a common use case of Machine Learning to classify the
sound types. For example, it can identify the bird species by their songs.

The Task Library `AudioClassifier` API can be used to deploy your custom audio
classifiers or pretrained ones into your mobile app.

## Key features of the AudioClassifier API

*   Input audio processing, e.g. converting PCM 16 bit encoding to PCM
    Float encoding and the manipulation of the audio ring buffer.

*   Label map locale.

*   Supporting Multi-head classification model.

*   Supporting both single-label and multi-label classification.

*   Score threshold to filter results.

*   Top-k classification results.

*   Label allowlist and denylist.

## Supported audio classifier models

The following models are guaranteed to be compatible with the `AudioClassifier`
API.

*   Models created by
    [TensorFlow Lite Model Maker for Audio Classification](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier).

*   The
    [pretrained audio event classification models on TensorFlow Hub](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1).

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

See the
[Audio Classification reference app](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)
for an example using `AudioClassifier` in an Android app.

### Step 1: Import Gradle dependency and other settings

Copy the `.tflite` model file to the assets directory of the Android module
where the model will be run. Specify that the file should not be compressed, and
add the TensorFlow Lite library to the module’s `build.gradle` file:

```java
android {
    // Other settings

    // Specify that the tflite file should not be compressed when building the APK package.
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // Other dependencies

    // Import the Audio Task Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.4.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
}
```

Note: starting from version 4.1 of the Android Gradle plugin, .tflite will be
added to the noCompress list by default and the above aaptOptions is not needed
anymore.

### Step 2: Using the model

```java
// Initialization
AudioClassifierOptions options =
    AudioClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
AudioClassifier classifier =
    AudioClassifier.createFromFileAndOptions(context, modelFile, options);

// Start recording
AudioRecord record = classifier.createAudioRecord();
record.startRecording();

// Load latest audio samples
TensorAudio audioTensor = classifier.createInputTensorAudio();
audioTensor.load(record);

// Run inference
List<Classifications> results = audioClassifier.classify(audioTensor);
```

See the
[source code and javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier.java)
for more options to configure `AudioClassifier`.

## Run inference in Python

### Step 1: Install the pip package

```
pip install tflite-support
```

Note: Task Library's Audio APIs rely on [PortAudio](http://www.portaudio.com/docs/v19-doxydocs/index.html)
to record audio from the device's microphone. If you intend to use Task
Library's [AudioRecord](/lite/api_docs/python/tflite_support/task/audio/AudioRecord)
for audio recording, you need to install PortAudio on your system.

* Linux: Run `sudo apt-get update && apt-get install libportaudio2`
* Mac and Windows: PortAudio is installed automatically when installing the
`tflite-support` pip package.

### Step 2: Using the model

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
classification_options = processor.ClassificationOptions(max_results=2)
options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = audio.AudioClassifier.create_from_options(options)

# Alternatively, you can create an audio classifier in the following manner:
# classifier = audio.AudioClassifier.create_from_file(model_path)

# Run inference
audio_file = audio.TensorAudio.create_from_wav_file(audio_path, classifier.required_input_buffer_size)
audio_result = classifier.classify(audio_file)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/audio/audio_classifier.py)
for more options to configure `AudioClassifier`.

## Run inference in C++

```c++
// Initialization
AudioClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<AudioClassifier> audio_classifier = AudioClassifier::CreateFromOptions(options).value();

// Create input audio buffer from data.
int input_buffer_size = audio_classifier->GetRequiredInputBufferSize();
const std::unique_ptr<AudioBuffer> audio_buffer =
    AudioBuffer::Create(audio_data.get(), input_buffer_size, kAudioFormat).value();

// Run inference
const ClassificationResult result = audio_classifier->Classify(*audio_buffer).value();
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/audio/audio_classifier.h)
for more options to configure `AudioClassifier`.

## Model compatibility requirements

The `AudioClassifier` API expects a TFLite model with mandatory
[TFLite Model Metadata](../../models/convert/metadata.md). See examples of creating
metadata for audio classifiers using the
[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#audio_classifiers).

The compatible audio classifier models should meet the following requirements:

*   Input audio tensor (kTfLiteFloat32)

    -   audio clip of size `[batch x samples]`.
    -   batch inference is not supported (`batch` is required to be 1).
    -   for multi-channel models, the channels need to be interleaved.

*   Output score tensor (kTfLiteFloat32)

    -   `[1 x N]` array with `N` represents the class number.
    -   optional (but recommended) label map(s) as AssociatedFile-s with type
        TENSOR_AXIS_LABELS, containing one label per line. The first such
        AssociatedFile (if any) is used to fill the `label` field (named as
        `class_name` in C++) of the results. The `display_name` field is filled
        from the AssociatedFile (if any) whose locale matches the
        `display_names_locale` field of the `AudioClassifierOptions` used at
        creation time ("en" by default, i.e. English). If none of these are
        available, only the `index` field of the results will be filled.
