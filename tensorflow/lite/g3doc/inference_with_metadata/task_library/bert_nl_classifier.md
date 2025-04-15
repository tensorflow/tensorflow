# Integrate BERT natural language classifier

The Task Library `BertNLClassifier` API is very similar to the `NLClassifier`
that classifies input text into different categories, except that this API is
specially tailored for Bert related models that require Wordpiece and
Sentencepiece tokenizations outside the TFLite model.

## Key features of the BertNLClassifier API

*   Takes a single string as input, performs classification with the string and
    outputs <Label, Score> pairs as classification results.

*   Performs out-of-graph
    [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h)
    or
    [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h)
    tokenizations on input text.

## Supported BertNLClassifier models

The following models are compatible with the `BertNLClassifier` API.

*   Bert Models created by
    [TensorFlow Lite Model Maker for text Classfication](https://ai.google.dev/edge/litert/libraries/modify/text_classification).

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

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

    // Import the Task Text Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.4'
}
```

Note: starting from version 4.1 of the Android Gradle plugin, .tflite will be
added to the noCompress list by default and the aaptOptions above is not needed
anymore.

### Step 2: Run inference using the API

```java
// Initialization
BertNLClassifierOptions options =
    BertNLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertNLClassifier classifier =
    BertNLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java)
for more details.

## Run inference in Swift

### Step 1: Import CocoaPods

Add the TensorFlowLiteTaskText pod in Podfile

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.4.4'
end
```

### Step 2: Run inference using the API

```swift
// Initialization
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLBertNLClassifier.h)
for more details.

## Run inference in C++

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_nl_classifier.h)
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

# Initialization
classifier = text.BertNLClassifier.create_from_file(model_path)

# Run inference
text_classification_result = classifier.classify(text)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/bert_nl_classifier.py)
for more options to configure `BertNLClassifier`.

## Example results

Here is an example of the classification results of movie reviews using the
[MobileBert](https://ai.google.dev/edge/litert/libraries/modify/text_classification)
model from Model Maker.

Input: "it's a charming and often affecting journey"

Output:

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

Try out the simple
[CLI demo tool for BertNLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier)
with your own model and test data.

## Model compatibility requirements

The `BetNLClassifier` API expects a TFLite model with mandatory
[TFLite Model Metadata](../../models/convert/metadata.md).

The Metadata should meet the following requirements:

*   input_process_units for Wordpiece/Sentencepiece Tokenizer

*   3 input tensors with names "ids", "mask" and "segment_ids" for the output of
    the tokenizer

*   1 output tensor of type float32, with a optionally attached label file. If a
    label file is attached, the file should be a plain text file with one label
    per line and the number of labels should match the number of categories as
    the model outputs.
