# Pose estimation

<img src="../images/pose.png" class="attempt-right" />

## Get started

_PoseNet_ is a vision model that can be used to estimate the pose of a person in
an image or video by estimating where key body joints are.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">
Download starter model</a>

Android and iOS end-to-end tutorials are coming soon. In the meantime, if you
want to experiment this on a web browser, check out the
<a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TensorFlow.js
GitHub repository</a>.

### Example applications and guides

There is a TensorFlow Lite sample application that demonstrates the PoseNet
model on Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android">
Android example</a>.

## How it works

Pose estimation refers to computer vision techniques that detect human figures
in images and videos, so that one could determine, for example, where someone’s
elbow shows up in an image.

To be clear, this technology is not recognizing who is in an image. The
algorithm is simply estimating where key body joints are.

The key points detected are indexed by "Part ID", with a confidence score
between 0.0 and 1.0, 1.0 being the highest.

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

## Performance Benchmarks

Performance benchmark numbers are generated with the tool
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
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">Posenet</a>
    </td>
    <td rowspan = 3>
      12.7 Mb
    </td>
    <td>Pixel 3 (Android 10) </td>
    <td>12ms</td>
    <td>31ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>12ms</td>
    <td>19ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1) </td>
     <td>4.8ms</td>
    <td>22ms** </td>
  </tr>
</table>

\* 4 threads used.

\*\* 2 threads used on iPhone for the best performance result.

## Example output

<img alt="Animation showing pose estimation" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif"/>

## How it performs

Performance varies based on your device and output stride (heatmaps and offset
vectors). The PoseNet model is image size invariant, which means it can predict
pose positions in the same scale as the original image regardless of whether the
image is downscaled. This means PoseNet can be configured to have a higher
accuracy at the expense of performance.

The output stride determines how much we’re scaling down the output relative to
the input image size. It affects the size of the layers and the model outputs.
The higher the output stride, the smaller the resolution of layers in the
network and the outputs, and correspondingly their accuracy. In this
implementation, the output stride can have values of 8, 16, or 32. In other
words, an output stride of 32 will result in the fastest performance but lowest
accuracy, while 8 will result in the highest accuracy but slowest performance.
We recommend starting with 16.

The following image shows how the output stride determines how much we’re
scaling down the output relative to the input image size. A higher output stride
is faster but results in lower accuracy.

<img alt="Output stride and heatmap resolution" src="../images/output_stride.png" >

## Read more about pose estimation

<ul>
  <li><a href="https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5">Blog post: Real-time Human Pose Estimation in the Browser with TensorFlow.js</a></li>
  <li><a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TF.js GitHub: Pose Detection in the Browser: PoseNet Model</a></li>
   <li><a href="https://medium.com/tensorflow/track-human-poses-in-real-time-on-android-with-tensorflow-lite-e66d0f3e6f9e">Blog post: Track human poses in real-time on Android with TensorFlow Lite</a></li>
</ul>

### Use cases

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></li>
  <li><a href="https://vimeo.com/34824490">Puppet Parade - Interactive Kinect Puppets</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce (Performance), Excerpts</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">Augmented reality</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">Interactive animation</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">Gait analysis</a></li>
</ul>
