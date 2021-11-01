# Pose estimation

<img src="../images/pose.png" class="attempt-right" />

Pose estimation is the task of using an ML model to estimate the pose of a
person from an image or a video by estimating the spatial locations of key body
joints (keypoints).

## Get started

If you are new to TensorFlow Lite and are working with Android or iOS, explore
the following example applications that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android">
Android example</a>
<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios">
iOS example</a>

If you are familiar with the
[TensorFlow Lite APIs](https://www.tensorflow.org/api_docs/python/tf/lite),
download the starter MoveNet pose estimation model and supporting files.

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet"> Download
starter model</a>

If you want to try pose estimation on a web browser, check out the
<a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet">
TensorFlow JS Demo</a>.

## Model description

### How it works

Pose estimation refers to computer vision techniques that detect human figures
in images and videos, so that one could determine, for example, where someone’s
elbow shows up in an image. It is important to be aware of the fact that pose
estimation merely estimates where key body joints are and does not recognize who
is in an image or video.

The pose estimation models takes a processed camera image as the input and
outputs information about keypoints. The keypoints detected are indexed by a
part ID, with a confidence score between 0.0 and 1.0. The confidence score
indicates the probability that a keypoint exists in that position.

We provides reference implementation of two TensorFlow Lite pose estimation
models:

*   MoveNet: the state-of-the-art pose estimation model available in two
    flavors: Lighting and Thunder. See a comparison between these two in the
    section below.
*   PoseNet: the previous generation pose estimation model released in 2017.

The various body joints detected by the pose estimation model are tabulated
below:

<table style="width: 30%;">
  <thead>
    <tr>
      <th>Id</th>
      <th>Part</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>nose</td>
    </tr>
    <tr>
      <td>1</td>
      <td>leftEye</td>
    </tr>
    <tr>
      <td>2</td>
      <td>rightEye</td>
    </tr>
    <tr>
      <td>3</td>
      <td>leftEar</td>
    </tr>
    <tr>
      <td>4</td>
      <td>rightEar</td>
    </tr>
    <tr>
      <td>5</td>
      <td>leftShoulder</td>
    </tr>
    <tr>
      <td>6</td>
      <td>rightShoulder</td>
    </tr>
    <tr>
      <td>7</td>
      <td>leftElbow</td>
    </tr>
    <tr>
      <td>8</td>
      <td>rightElbow</td>
    </tr>
    <tr>
      <td>9</td>
      <td>leftWrist</td>
    </tr>
    <tr>
      <td>10</td>
      <td>rightWrist</td>
    </tr>
    <tr>
      <td>11</td>
      <td>leftHip</td>
    </tr>
    <tr>
      <td>12</td>
      <td>rightHip</td>
    </tr>
    <tr>
      <td>13</td>
      <td>leftKnee</td>
    </tr>
    <tr>
      <td>14</td>
      <td>rightKnee</td>
    </tr>
    <tr>
      <td>15</td>
      <td>leftAnkle</td>
    </tr>
    <tr>
      <td>16</td>
      <td>rightAnkle</td>
    </tr>
  </tbody>
</table>

An example output is shown below:

<img alt="Animation showing pose estimation" src="https://storage.googleapis.com/download.tensorflow.org/example_images/movenet_demo.gif"/>

## Performance benchmarks

MoveNet is available in two flavors:

*   MoveNet.Lightning is smaller, faster but less accurate than the Thunder
    version. It can run in realtime on modern smartphones.
*   MoveNet.Thunder is the more accurate version but also larger and slower than
    Lightning. It is useful for the use cases that require higher accuracy.

MoveNet outperforms PoseNet on a variety of datasets, especially in images with
fitness action images. Therefore, we recommend using MoveNet over PoseNet.

Performance benchmark numbers are generated with the tool
[described here](../../performance/measurement). Accuracy (mAP) numbers are
measured on a subset of the [COCO dataset](https://cocodataset.org/#home) in
which we filter and crop each image to contain only one person .

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Size (MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">Latency (ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - CPU 4 threads</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - CPU 4 threads</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder (FP16 quantized)</a>
    </td>
    <td>12.6MB</td>
    <td>72.0</td>
    <td>155ms</td>
    <td>45ms</td>
    <td>594ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder (INT8 quantized)</a>
    </td>
    <td>7.1MB</td>
    <td>68.9</td>
    <td>100ms</td>
    <td>52ms</td>
    <td>251ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning (FP16 quantized)</a>
    </td>
    <td>4.8MB</td>
    <td>63.0</td>
    <td>60ms</td>
    <td>25ms</td>
    <td>186ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning (INT8 quantized)</a>
    </td>
    <td>2.9MB</td>
    <td>57.4</td>
    <td>52ms</td>
    <td>28ms</td>
    <td>95ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet(MobileNetV1 backbone, FP32)</a>
    </td>
    <td>13.3MB</td>
    <td>45.6</td>
    <td>80ms</td>
    <td>40ms</td>
    <td>338ms</td>
  </tr>
</tbody>
</table>

## Further reading and resources

*   Check out this
    [blog post](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html)
    to learn more about pose estimation using MoveNet and TensorFlow Lite.
*   Check out this
    [blog post](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)
    to learn more about pose estimation on the web.
*   Check out this [tutorial](https://www.tensorflow.org/hub/tutorials/movenet)
    to learn about running MoveNet on Python using a model from TensorFlow Hub.
*   Coral/EdgeTPU can make pose estimation run much faster on IoT devices. See
    [EdgeTPU-optimized models](https://coral.ai/models/pose-estimation/) for
    more details.
*   Read the PoseNet paper [here](https://arxiv.org/abs/1803.08225)

Also, check out these use cases of pose estimation.

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></li>
  <li><a href="https://vimeo.com/34824490">Puppet Parade - Interactive Kinect Puppets</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce (Performance), Excerpts</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">Augmented reality</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">Interactive animation</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">Gait analysis</a></li>
</ul>
