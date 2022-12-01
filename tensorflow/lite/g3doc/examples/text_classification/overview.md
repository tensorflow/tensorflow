# Text classification

Use a TensorFlow Lite model to category a paragraph into predefined groups.

Note: (1) To integrate an existing model, try
[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier).
(2) To customize a model, try
[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

## Get started

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend
exploring the guide of
[TensorFLow Lite Task Library](../../inference_with_metadata/task_library/nl_classifier)
to integrate text classification models within just a few lines of code. You can
also integrate the model using the
[TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java).

The Android example below demonstrates the implementation for both methods as
[lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_task_api)
and
[lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter),
respectively.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android
example</a>

If you are using a platform other than Android, or you are already familiar with
the TensorFlow Lite APIs, you can download our starter text classification
model.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Download
starter model</a>

## How it works

Text classification categorizes a paragraph into predefined groups based on its
content.

This pretrained model predicts if a paragraph's sentiment is positive or
negative. It was trained on
[Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/)
from Mass et al, which consists of IMDB movie reviews labeled as either positive
or negative.

Here are the steps to classify a paragraph with the model:

1.  Tokenize the paragraph and convert it to a list of word ids using a
    predefined vocabulary.
1.  Feed the list to the TensorFlow Lite model.
1.  Get the probability of the paragraph being positive or negative from the
    model outputs.

### Note

*   Only English is supported.
*   This model was trained on movie reviews dataset so you may experience
    reduced accuracy when classifying text of other domains.

## Performance benchmarks

Performance benchmark numbers are generated with the tool
[described here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>Device </th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 3>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Text Classification</a>
    </td>
    <td rowspan = 3>
      0.6 Mb
    </td>
    <td>Pixel 3 (Android 10) </td>
    <td>0.05ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>0.05ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1) </td>
    <td>0.025ms** </td>
  </tr>
</table>

\* 4 threads used.

\*\* 2 threads used on iPhone for the best performance result.

## Example output

| Text                                       | Negative (0) | Positive (1) |
| ------------------------------------------ | ------------ | ------------ |
| This is the best movie I’ve seen in recent | 25.3%        | 74.7%        |
: years. Strongly recommend it!              :              :              :
| What a waste of my time.                   | 72.5%        | 27.5%        |

## Use your training dataset

Follow this
[tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification)
to apply the same technique used here to train a text classification model using
your own datasets. With the right dataset, you can create a model for use cases
such as document categorization or toxic comments detection.

## Read more about text classification

*   [Word embeddings and tutorial to train this model](https://www.tensorflow.org/tutorials/text/word_embeddings)
