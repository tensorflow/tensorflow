# Pose estimation
<img src="../images/pose.png" class="attempt-right" />

<i>PoseNet</i> is a vision model that can be used to estimate the pose of a person in an image/video by estimating where key body joints are.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite">Download starter model</a>

## Tutorials (coming soon)
<a class="button button-primary" href="">iOS</a>
<a class="button button-primary" href="">Android</a>

## How it works
Pose estimation refers to computer vision techniques that detect human figures in images and videos, so that one could determine, for example, where someone’s elbow shows up in an image.

To be clear, this technology is not recognizing who is in an image — there is no personal identifiable information associated to pose detection. The algorithm is simply estimating where key body joints are.

The key points detected are indexed by part id with a confidence score between 0.0 and 1.0; 1.0 being the highest.

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

## Example output
<img src="https://www.tensorflow.org/images/models/pose_estimation.gif" />

## Get started
Android and iOS end-to-end tutorials are coming soon. In the meantime, if you want to experiment this on a web browser, check out the TensorFlow.js <a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">GitHub repository</a>.


## How it performs
Performance varies based on your device and output stride (heatmaps and offset vectors). The PoseNet model is image size invariant, which means it can predict pose positions in the same scale as the original image regardless of whether the image is downscaled. This means PoseNet can be configured to have a higher accuracy at the expense of performance.

The output stride determines how much we’re scaling down the output relative to the input image size. It affects the size of the layers and the model outputs. The higher the output stride, the smaller the resolution of layers in the network and the outputs, and correspondingly their accuracy. In this implementation, the output stride can have values of 8, 16, or 32. In other words, an output stride of 32 will result in the fastest performance but lowest accuracy, while 8 will result in the highest accuracy but slowest performance. We recommend starting with 16.

<img src="../images/models/output_stride.png" >
<span style="font-size: 0.8em">The output stride determines how much we’re scaling down the output relative to the input image size. A higher output stride is faster but results in lower accuracy.</span>

## Read more about this
<ul>
  <li><a href="">Blog post: Real-time Human Pose Estimation in the Browser with TensorFlow.js</a></li>
  <li><a href="">TF.js GitHub: Pose Detection in the Browser: PoseNet Model</a></li>
</ul>

## Users
<ul>
  <li><a href="">‘PomPom Mirror’</a></li>
  <li><a href="">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></li>
  <li><a href="">Puppet Parade - Interactive Kinect Puppets</a></li>
  <li><a href="">Messa di Voce (Performance), Excerpts</a></li>
  <li><a href="">Augmented reality</a></li>
  <li><a href="">Interactive animation</a></li>
  <li><a href="">Gait analysis</a></li>
</ul>
