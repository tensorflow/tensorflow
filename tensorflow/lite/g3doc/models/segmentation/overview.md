# Segmentation

<img src="../images/segmentation.png" class="attempt-right" />

## Get started

_DeepLab_ is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g. person, dog, cat) to every
pixel in the input image.

If you are new to TensorFlow Lite and are working with Android or iOS, we
recommend exploring the following example applications that can help you get
started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Android
example</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">iOS
example</a>

If you are using a platform other than Android or iOS, or you are already
familiar with the
<a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite
APIs</a>, you can download our starter image segmentation model.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite">Download
starter model</a>

## How it works

Semantic image segmentation predicts whether each pixel of an image is
associated with a certain class. This is in contrast to
<a href="../object_detection/overview.md">object detection</a>, which detects
objects in rectangular regions, and
<a href="../image_classification/overview.md">image classification</a>, which
classifies the overall image.

The current implementation includes the following features:
<ol>
  <li>DeepLabv1: We use atrous convolution to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks.</li>
  <li>DeepLabv2: We use atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales with filters at multiple sampling rates and effective fields-of-views.</li>
  <li>DeepLabv3: We augment the ASPP module with image-level feature [5, 6] to capture longer range information. We also include batch normalization [7] parameters to facilitate the training. In particular, we applying atrous convolution to extract output features at different output strides during training and evaluation, which efficiently enables training BN at output stride = 16 and attains a high performance at output stride = 8 during evaluation.</li>
  <li>DeepLabv3+: We extend DeepLabv3 to include a simple yet effective decoder module to refine the segmentation results especially along object boundaries. Furthermore, in this encoder-decoder structure one can arbitrarily control the resolution of extracted encoder features by atrous convolution to trade-off precision and runtime.</li>
</ol>

## Performance benchmarks

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
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite">Deeplab v3</a>
    </td>
    <td rowspan = 3>
      2.7 Mb
    </td>
    <td>Pixel 3 (Android 10) </td>
    <td>16ms</td>
    <td>37ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>20ms</td>
    <td>23ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1) </td>
     <td>16ms</td>
    <td>25ms** </td>
  </tr>
</table>

\* 4 threads used.

\*\* 2 threads used on iPhone for the best performance result.

## Example output

The model will create a mask over the target objects with high accuracy.

<img alt="Animation showing image segmentation" src="images/segmentation.gif" />

## Read more about segmentation

<ul>
  <li><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">Semantic Image Segmentation with DeepLab in TensorFlow</a></li>
  <li><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">TensorFlow Lite Now Faster with Mobile GPUs (Developer Preview)</a></li>
  <li><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab: Deep Labelling for Semantic Image Segmentation</a></li>
</ul>
