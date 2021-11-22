# Integrate BERT question answerer

The Task Library `BertQuestionAnswerer` API loads a Bert model and answers
questions based on the content of a given passage. For more information, see the
documentation for the Question-Answer model
<a href="../../models/bert_qa/overview.md">here</a>.

## Key features of the BertQuestionAnswerer API

*   Takes two text inputs as question and context and outputs a list of possible
    answers.

*   Performs out-of-graph Wordpiece or Sentencepiece tokenizations on input
    text.

## Supported BertQuestionAnswerer models

The following models are compatible with the `BertNLClassifier` API.

*   Models created by
    [TensorFlow Lite Model Maker for BERT Question Answer](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer).

*   The
    [pretrained BERT models on TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1).

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
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
}
```

Note: starting from version 4.1 of the Android Gradle plugin, .tflite will be
added to the noCompress list by default and the aaptOptions above is not needed
anymore.

### Step 2: Run inference using the API

```java
// Initialization
BertQuestionAnswererOptions options =
    BertQuestionAnswererOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertQuestionAnswerer answerer =
    BertQuestionAnswerer.createFromFileAndOptions(
        androidContext, modelFile, options);

// Run inference
List<QaAnswer> answers = answerer.answer(contextOfTheQuestion, questionToAsk);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java)
for more details.

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
let mobileBertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(
      modelPath: mobileBertModelPath)

// Run inference
let answers = mobileBertAnswerer.answer(
      context: context, question: question)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h)
for more details.

## Run inference in C++

```c++
// Initialization
BertQuestionAnswererOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<BertQuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference
std::vector<QaAnswer> positive_results = answerer->Answer(context_of_question, question_to_ask);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h)
for more details.

## Example results

Here is an example of the answer results of
[ALBERT model](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1).

Context: "The Amazon rainforest, alternatively, the Amazon Jungle, also known in
English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon
biome that covers most of the Amazon basin of South America. This basin
encompasses 7,000,000 km2 (2,700,000 sq mi), of which
5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region
includes territory belonging to nine nations."

Question: "Where is Amazon rainforest?"

Answers:

```
answer[0]:  'South America.'
logit: 1.84847, start_index: 39, end_index: 40
answer[1]:  'most of the Amazon basin of South America.'
logit: 1.2921, start_index: 34, end_index: 40
answer[2]:  'the Amazon basin of South America.'
logit: -0.0959535, start_index: 36, end_index: 40
answer[3]:  'the Amazon biome that covers most of the Amazon basin of South America.'
logit: -0.498558, start_index: 28, end_index: 40
answer[4]:  'Amazon basin of South America.'
logit: -0.774266, start_index: 37, end_index: 40

```

Try out the simple
[CLI demo tool for BertQuestionAnswerer](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer)
with your own model and test data.

## Model compatibility requirements

The `BertQuestionAnswerer` API expects a TFLite model with mandatory
[TFLite Model Metadata](../../convert/metadata.md).

The Metadata should meet the following requirements:

*   `input_process_units` for Wordpiece/Sentencepiece Tokenizer

*   3 input tensors with names "ids", "mask" and "segment_ids" for the output of
    the tokenizer

*   2 output tensors with names "end_logits" and "start_logits" to indicate the
    answer's relative position in the context
