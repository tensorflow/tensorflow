# Object detection

Given an image or a video stream, an object detection model can identify which
of a known set of objects might be present and provide information about their
positions within the image.

For example, this screenshot of the <a href="#get_started">example
application</a> shows how two objects have been recognized and their positions
annotated:

<img src="images/android_apple_banana.png" alt="Screenshot of Android example" width="30%">

Note: (1) To integrate an existing model, try
[TensorFlow Lite Task Library](https://ai.google.dev/edge/litert/libraries/task_library/object_detector).
(2) To customize a model, try
[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker).

## Get started

To learn how to use object detection in a mobile app, explore the
<a href="#example_applications_and_guides">Example applications and guides</a>.

If you are using a platform other than Android or iOS, or if you are already
familiar with the
<a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite
APIs</a>, you can download our starter object detection model and the
accompanying labels.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">Download
starter model with Metadata</a>

For more information about Metadata and associated fields (eg: `labels.txt`) see
<a href="../../models/convert/metadata#read_the_metadata_from_models">Read
the metadata from models</a>

If you want to train a custom detection model for your own task, see
<a href="#model-customization">Model customization</a>.

For the following use cases, you should use a different type of model:

<ul>
  <li>Predicting which single label the image most likely represents (see <a href="../image_classification/overview.md">image classification</a>)</li>
  <li>Predicting the composition of an image, for example subject versus background (see <a href="../segmentation/overview.md">segmentation</a>)</li>
</ul>

### Example applications and guides

If you are new to TensorFlow Lite and are working with Android or iOS, we
recommend exploring the following example applications that can help you get
started.

#### Android

You can leverage the out-of-box API from
[TensorFlow Lite Task Library](https://ai.google.dev/edge/litert/libraries/task_library/object_detector)
to integrate object detection models in just a few lines of code. You can also
build your own custom inference pipeline using the
[TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java).

The Android example below demonstrates the implementation for both methods
using
[Task library](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services)
and
[interpreter API](https://github.com/tensorflow/examples/tree/eb925e460f761f5ed643d17f0c449e040ac2ac45/lite/examples/object_detection/android/lib_interpreter),
respectively.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">View
Android example</a>

#### iOS

You can integrate the model using the
[TensorFlow Lite Interpreter Swift API](../../guide/inference#load_and_run_a_model_in_swift).
See the iOS example below.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios">View
iOS example</a>

## Model description

This section describes the signature for
[Single-Shot Detector](https://arxiv.org/abs/1512.02325) models converted to
TensorFlow Lite from the
[TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/).

An object detection model is trained to detect the presence and location of
multiple classes of objects. For example, a model might be trained with images
that contain various pieces of fruit, along with a _label_ that specifies the
class of fruit they represent (e.g. an apple, a banana, or a strawberry), and
data specifying where each object appears in the image.

When an image is subsequently provided to the model, it will output a list of
the objects it detects, the location of a bounding box that contains each
object, and a score that indicates the confidence that detection was correct.

### Input Signature

The model takes an image as input.

Lets assume the expected image is 300x300 pixels, with three channels (red,
blue, and green) per pixel. This should be fed to the model as a flattened
buffer of 270,000 byte values (300x300x3). If the model is
<a href="../../performance/post_training_quantization.md">quantized</a>, each
value should be a single byte representing a value between 0 and 255.

You can take a look at our
[example app code](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)
to understand how to do this pre-processing on Android.

### Output Signature

The model outputs four arrays, mapped to the indices 0-4. Arrays 0, 1, and 2
describe `N` detected objects, with one element in each array corresponding to
each object.

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Locations</td>
      <td>Multidimensional array of [N][4] floating point values between 0 and 1, the inner arrays representing bounding boxes in the form [top, left, bottom, right]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Classes</td>
      <td>Array of N integers (output as floating point values) each indicating the index of a class label from the labels file</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Scores</td>
      <td>Array of N floating point values between 0 and 1 representing probability that a class was detected</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Number of detections</td>
      <td>Integer value of N</td>
    </tr>
  </tbody>
</table>

NOTE: The number of results (10 in the above case) is a parameter set while
exporting the detection model to TensorFlow Lite. See
<a href="#model-customization">Model customization</a> for more details.

For example, imagine a model has been trained to detect apples, bananas, and
strawberries. When provided an image, it will output a set number of detection
results - in this example, 5.

<table style="width: 60%;">
  <thead>
    <tr>
      <th>Class</th>
      <th>Score</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apple</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Strawberry</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163] </td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0.23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>0.11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### Confidence score

To interpret these results, we can look at the score and the location for each
detected object. The score is a number between 0 and 1 that indicates confidence
that the object was genuinely detected. The closer the number is to 1, the more
confident the model is.

Depending on your application, you can decide a cut-off threshold below which
you will discard detection results. For the current example, a sensible cut-off
is a score of 0.5 (meaning a 50% probability that the detection is valid). In
that case, the last two objects in the array would be ignored because those
confidence scores are below 0.5:

<table style="width: 60%;">
  <thead>
    <tr>
      <th>Class</th>
      <th>Score</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apple</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Strawberry</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163] </td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Banana</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Apple</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

The cut-off you use should be based on whether you are more comfortable with
false positives (objects that are wrongly identified, or areas of the image that
are erroneously identified as objects when they are not), or false negatives
(genuine objects that are missed because their confidence was low).

For example, in the following image, a pear (which is not an object that the
model was trained to detect) was misidentified as a "person". This is an example
of a false positive that could be ignored by selecting an appropriate cut-off.
In this case, a cut-off of 0.6 (or 60%) would comfortably exclude the false
positive.

<img src="images/false_positive.png" alt="Screenshot of Android example showing a false positive" width="30%">

#### Location

For each detected object, the model will return an array of four numbers
representing a bounding rectangle that surrounds its position. For the starter
model provided, the numbers are ordered as follows:

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>top,</td>
      <td>left,</td>
      <td>bottom,</td>
      <td>right</td>
      <td>]</td>
    </tr>
  </tbody>
</table>

The top value represents the distance of the rectangle’s top edge from the top
of the image, in pixels. The left value represents the left edge’s distance from
the left of the input image. The other values represent the bottom and right
edges in a similar manner.

Note: Object detection models accept input images of a specific size. This is likely to be different from the size of the raw image captured by your device’s camera, and you will have to write code to crop and scale your raw image to fit the model’s input size (there are examples of this in our <a href="#get_started">example applications</a>).<br /><br />The pixel values output by the model refer to the position in the cropped and scaled image, so you must scale them to fit the raw image in order to interpret them correctly.

## Performance benchmarks

Performance benchmark numbers for our
<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">starter
model</a> are generated with the tool
[described here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>Device </th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 3>
      <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
    </td>
    <td rowspan = 3>
      27 Mb
    </td>
    <td>Pixel 3 (Android 10) </td>
    <td>22ms</td>
    <td>46ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>20ms</td>
    <td>29ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1) </td>
     <td>7.6ms</td>
    <td>11ms** </td>
  </tr>
</table>

\* 4 threads used.

\*\* 2 threads used on iPhone for the best performance result.

## Model Customization

### Pre-trained models

Mobile-optimized detection models with a variety of latency and precision
characteristics can be found in the
[Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models).
Each one of them follows the input and output signatures described in the
following sections.

Most of the download zips contain a `model.tflite` file. If there isn't one, a
TensorFlow Lite flatbuffer can be generated using
[these instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md).
SSD models from the
[TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
can also be converted to TensorFlow Lite using the instructions
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md).
It is important to note that detection models cannot be converted directly using
the [TensorFlow Lite Converter](../../models/convert), since
they require an intermediate step of generating a mobile-friendly source model.
The scripts linked above perform this step.

Both the
[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)
&
[TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)
exporting scripts have parameters that can enable a larger number of output
objects or slower, more-accurate post processing. Please use `--help` with the
scripts to see an exhaustive list of supported arguments.

> Currently, on-device inference is only optimized with SSD models. Better
> support for other architectures like CenterNet and EfficientDet is being
> investigated.

### How to choose a model to customize?

Each model comes with its own precision (quantified by mAP value) and latency
characteristics. You should choose a model that works the best for your use-case
and intended hardware. For example, the
[Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models)
models are ideal for inference on Google's Edge TPU on Pixel 4.

You can use our
[benchmark tool](https://www.tensorflow.org/lite/performance/measurement) to
evaluate models and choose the most efficient option available.

## Fine-tuning models on custom data

The pre-trained models we provide are trained to detect 90 classes of objects.
For a full list of classes, see the labels file in the
<a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">model
metadata</a>.

You can use a technique known as transfer learning to re-train a model to
recognize classes not in the original set. For example, you could re-train the
model to detect multiple types of vegetable, despite there only being one
vegetable in the original training data. To do this, you will need a set of
training images for each of the new labels you wish to train. The recommended
way is to use
[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)
library which simplifies the process of training a TensorFlow Lite model using
custom dataset, with a few lines of codes. It uses transfer learning to reduce
the amount of required training data and time. You can also learn from
[Few-shot detection Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb)
as an example of fine-tuning a pre-trained model with few examples.

For fine-tuning with larger datasets, take a look at the these guides for
training your own models with the TensorFlow Object Detection API:
[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md),
[TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md).
Once trained, they can be converted to a TFLite-friendly format with the
instructions here:
[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md),
[TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)
