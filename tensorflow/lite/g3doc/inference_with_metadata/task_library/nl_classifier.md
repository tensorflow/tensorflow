# Integrate Natural language classifier

The Task Library's `NLClassifier` API classifies input text into different
categories, and is a versatile and configurable API that can handle most text
classification models.

## Key features of the NLClassifier API

*   Takes a single string as input, performs classification with the string and
    outputs <Label, Score> pairs as classification results.

*   Optional Regex Tokenization available for input text.

*   Configurable to adapt different classification models.

## Supported NLClassifier models

The following models are guaranteed to be compatible with the `NLClassifier`
API.

*   The <a href="../../examples/text_classification/overview">movie review
    sentiment classification</a> model.

*   Models with `average_word_vec` spec created by
    [TensorFlow Lite Model Maker for text Classification](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

See the
[Text Classification reference app](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/textclassification/client/TextClassificationClient.java)
for an example of how to use `NLClassifier` in an Android app.

### Step 1: Import Gradle dependency and other settings

Copy the `.tflite` model file to the assets directory of the Android module
where the model will be run. Specify that the file should not be compressed, and
add the TensorFlow Lite library to the module’s `build.gradle` file:

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.3.0'
}
```

Note: starting from version 4.1 of the Android Gradle plugin, .tflite will be
added to the noCompress list by default and the aaptOptions above is not needed
anymore.

### Step 2: Run inference using the API

```java
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options =
    NLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setInputTensorName(INPUT_TENSOR_NAME)
        .setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME)
        .build();
NLClassifier classifier =
    NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier.java)
for more options to configure `NLClassifier`.

## Run inference in Swift

### Step 1: Import CocoaPods

Add the TensorFlowLiteTaskText pod in Podfile

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.2.0'
end
```

### Step 2: Run inference using the API

```swift
// Initialization
var modelOptions:TFLNLClassifierOptions = TFLNLClassifierOptions()
modelOptions.inputTensorName = inputTensorName
modelOptions.outputScoreTensorName = outputScoreTensorName
let nlClassifier = TFLNLClassifier.nlClassifier(
      modelPath: modelPath,
      options: modelOptions)

// Run inference
let categories = nlClassifier.classify(text: input)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLNLClassifier.h)
for more details.

## Run inference in C++

```c++
// Initialization
NLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<NLClassifier> classifier = NLClassifier::CreateFromOptions(options).value();

// Run inference
std::vector<core::Category> categories = classifier->Classify(kInput);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h)
for more details.

## Run inference in Python

### Step 1: Install the pip package

```
pip install tflite-support
```

### Step 2: Using the model

```python
# Imports
from tflite_support.task import text
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
options = text.NLClassifierOptions(base_options=base_options)
classifier = text.NLClassifier.create_from_options(options)

# Alternatively, you can create an NLClassifier in the following manner:
# classifier = text.NLClassifier.create_from_file(model_path)

# Run inference
text_classification_result = classifier.classify(text)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/nl_classifier.py)
for more options to configure `NLClassifier`.

## Example results

Here is an example of the classification results of the
[movie review model](https://www.tensorflow.org/lite/examples/text_classification/overview).

Input: "What a waste of my time."

Output:

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

Try out the simple
[CLI demo tool for NLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier)
with your own model and test data.

## Model compatibility requirements

Depending on the use case, the `NLClassifier` API can load a TFLite model with
or without [TFLite Model Metadata](../../models/convert/metadata). See examples of
creating metadata for natural language classifiers using the
[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#nl_classifiers).

The compatible models should meet the following requirements:

*   Input tensor: (kTfLiteString/kTfLiteInt32)

    -   Input of the model should be either a kTfLiteString tensor raw input
        string or a kTfLiteInt32 tensor for regex tokenized indices of raw input
        string.
    -   If input type is kTfLiteString, no [Metadata](../../models/convert/metadata)
        is required for the model.
    -   If input type is kTfLiteInt32, a `RegexTokenizer` needs to be set up in
        the input tensor's
        [Metadata](https://www.tensorflow.org/lite/models/convert/metadata_writer_tutorial#natural_language_classifiers).

*   Output score tensor:
    (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    -   Mandatory output tensor for the score of each category classified.

    -   If type is one of the Int types, dequantize it to double/float to
        corresponding platforms

    -   Can have an optional associated file in the output tensor's
        corresponding [Metadata](../../models/convert/metadata) for category labels,
        the file should be a plain text file with one label per line, and the
        number of labels should match the number of categories as the model
        outputs. See the
        [example label file](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/nl_classifier/labels.txt).

*   Output label tensor: (kTfLiteString/kTfLiteInt32)

    -   Optional output tensor for the label for each category, should be of the
        same length as the output score tensor. If this tensor is not present,
        the API uses score indices as classnames.

    -   Will be ignored if the associated label file is present in output score
        tensor's Metadata.

