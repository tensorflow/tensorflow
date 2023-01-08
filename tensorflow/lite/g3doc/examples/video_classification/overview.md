# Video classification

<img src="../images/video.png" class="attempt-right">

*Video classification* is the machine learning task of identifying what a video
represents. A video classification model is trained on a video dataset that
contains a set of unique classes, such as different actions or movements. The
model receives video frames as input and outputs the probability of each class
being represented in the video.

Video classification and image classification models both use images as inputs
to predict the probabilities of those images belonging to predefined classes.
However, a video classification model also processes the spatio-temporal
relationships between adjacent frames to recognize the actions in a video.

For example, a *video action recognition* model can be trained to identify human
actions like running, clapping, and waving. The following image shows the output
of a video classification model on Android.

<img alt="Screenshot of Android example" src="https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/push-up-classification.gif"/>

## Get started

If you are using a platform other than Android or Raspberry Pi, or if you are
already familiar with the
[TensorFlow Lite APIs](https://www.tensorflow.org/api_docs/python/tf/lite),
download the starter video classification model and the supporting files. You
can also build your own custom inference pipeline using the
[TensorFlow Lite Support Library](../../inference_with_metadata/lite_support).

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/movinet/a0/stream/kinetics-600/classification/tflite/int8/1">Download
starter model with metadata</a>

If you are new to TensorFlow Lite and are working with Android or Raspberry Pi,
explore the following example applications to help you get started.

### Android

The Android application uses the device's back camera for continuous video
classification. Inference is performed using the
[TensorFlow Lite Java API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/package-summary).
The demo app classifies frames and displays the predicted classifications in
real time.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/android">Android
example</a>

### Raspberry Pi

The Raspberry Pi example uses TensorFlow Lite with Python to perform continuous
video classification. Connect the Raspberry Pi to a camera, like Pi Camera, to
perform real-time video classification. To view results from the camera, connect
a monitor to the Raspberry Pi and use SSH to access the Pi shell (to avoid
connecting a keyboard to the Pi).

Before starting,
[set up](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
your Raspberry Pi with Raspberry Pi OS (preferably updated to Buster).

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/raspberry_pi ">Raspberry
Pi example</a>

## Model description

Mobile Video Networks
([MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet))
are a family of efficient video classification models optimized for mobile
devices. MoViNets demonstrate state-of-the-art accuracy and efficiency on
several large-scale video action recognition datasets, making them well-suited
for *video action recognition* tasks.

There are three variants of the
[MoviNet](https://tfhub.dev/s?deployment-format=lite&q=movinet) model for
TensorFlow Lite:
[MoviNet-A0](https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification),
[MoviNet-A1](https://tfhub.dev/tensorflow/movinet/a1/stream/kinetics-600/classification),
and
[MoviNet-A2](https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification).
These variants were trained with the
[Kinetics-600](https://arxiv.org/abs/1808.01340) dataset to recognize 600
different human actions. *MoviNet-A0* is the smallest, fastest, and least
accurate. *MoviNet-A2* is the largest, slowest, and most accurate. *MoviNet-A1*
is a compromise between A0 and A2.

### How it works

During training, a video classification model is provided videos and their
associated *labels*. Each label is the name of a distinct concept, or class,
that the model will learn to recognize. For *video action recognition*, the
videos will be of human actions and the labels will be the associated action.

The video classification model can learn to predict whether new videos belong to
any of the classes provided during training. This process is called *inference*.
You can also use
[transfer learning](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
to identify new classes of videos by using a pre-existing model.

The model is a streaming model that receives continuous video and responds in
real time. As the model receives a video stream, it identifies whether any of
the classes from the training dataset are represented in the video. For each
frame, the model returns these classes, along with the probability that the
video represents the class. An example output at a given time might look as
follows:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Action</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>square dancing</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>threading needle</td>
      <td>0.08</td>
    </tr>
    <tr>
      <td>twiddling fingers</td>
      <td>0.23</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Waving hand</td>
      <td style="background-color: #fcb66d;">0.67</td>
    </tr>
  </tbody>
</table>

Each action in the output corresponds to a label in the training data. The
probability denotes the likelihood that the action is being displayed in the
video.

### Model inputs

The model accepts a stream of RGB video frames as input. The size of the input
video is flexible, but ideally it matches the model training resolution and
frame-rate:

*   **MoviNet-A0**: 172 x 172 at 5 fps
*   **MoviNet-A1**: 172 x 172 at 5 fps
*   **MoviNet-A1**: 224 x 224 at 5 fps

The input videos are expected to have color values within the range of 0 and 1,
following the common
[image input conventions](https://www.tensorflow.org/hub/common_signatures/images#input).

Internally, the model also analyzes the context of each frame by using
information gathered in previous frames. This is accomplished by taking internal
states from the model output and feeding it back into the model for upcoming
frames.

### Model outputs

The model returns a series of labels and their corresponding scores. The scores
are logit values that represent the prediction for each class. These scores can
be converted to probabilities by using the softmax function (`tf.nn.softmax`).

```python
    exp_logits = np.exp(np.squeeze(logits, axis=0))
    probabilities = exp_logits / np.sum(exp_logits)
```

Internally, the model output also includes internal states from the model and
feeds it back into the model for upcoming frames.

## Performance benchmarks

Performance benchmark numbers are generated with the
[benchmarking tool](https://www.tensorflow.org/lite/performance/measurement).
MoviNets only support CPU.

Model performance is measured by the amount of time it takes for a model to run
inference on a given piece of hardware. A lower time implies a faster model.
Accuracy is measured by how often the model correctly classifies a class in a
video.

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Size </th>
      <th>Accuracy *</th>
      <th>Device</th>
      <th>CPU **</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 2>
MoviNet-A0 (Integer quantized)
    </td>
    <td rowspan = 2>
      3.1 MB
    </td>
    <td rowspan = 2>65%</td>
    <td>Pixel 4</td>
    <td>5 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>11 ms</td>
  </tr>
    <tr>
    <td rowspan = 2>
MoviNet-A1 (Integer quantized)
    </td>
    <td rowspan = 2>
      4.5 MB
    </td>
    <td rowspan = 2>70%</td>
    <td>Pixel 4</td>
    <td>8 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>19 ms</td>
  </tr>
      <tr>
    <td rowspan = 2>
MoviNet-A2 (Integer quantized)
    </td>
    <td rowspan = 2>
      5.1 MB
    </td>
    <td rowspan = 2>72%</td>
    <td>Pixel 4</td>
    <td>15 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>36 ms</td>
  </tr>
</table>

\* Top-1 accuracy measured on the
[Kinetics-600](https://arxiv.org/abs/1808.01340) dataset.

\*\* Latency measured when running on CPU with 1-thread.

## Model customization

The pre-trained models are trained to recognize 600 human actions from the
[Kinetics-600](https://arxiv.org/abs/1808.01340) dataset. You can also use
transfer learning to re-train a model to recognize human actions that are not in
the original set. To do this, you need a set of training videos for each of the
new actions you want to incorporate into the model.

For more on fine-tuning models on custom data, see the
[MoViNets repo](https://github.com/tensorflow/models/tree/master/official/projects/movinet)
and
[MoViNets tutorial](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb).

## Further reading and resources

Use the following resources to learn more about concepts discussed on this page:

*   [MoViNets repo](https://github.com/tensorflow/models/tree/master/official/projects/movinet)
*   [MoViNets paper](https://arxiv.org/abs/2103.11511)
*   [Pretrained MoViNet models](https://tfhub.dev/s?deployment-format=lite&q=movinet)
*   [MoViNets tutorial](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
*   [Kinetics datasets](https://deepmind.com/research/open-source/kinetics)
