# Segmentation (GPU)
<img src="../images/segmentation.png" class="attempt-right" />

<i>DeepLab</i> is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite">Download starter model</a>

## Tutorials (coming soon)
<a class="button button-primary" href="">iOS</a>
<a class="button button-primary" href="">Android</a>

## How it works
It all started with classification where the model predicts an entire input. With advances in data, hardware, and software, object detection can infer objects with spatial location. Semantic segmentation offers the highest level of granularity with labels at a pixel level.

Current implementation includes the following features:
<ol>
  <li>DeepLabv1: We use atrous convolution to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks.</li>
  <li>DeepLabv2: We use atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales with filters at multiple sampling rates and effective fields-of-views.</li>
  <li>DeepLabv3: We augment the ASPP module with image-level feature [5, 6] to capture longer range information. We also include batch normalization [7] parameters to facilitate the training. In particular, we applying atrous convolution to extract output features at different output strides during training and evaluation, which efficiently enables training BN at output stride = 16 and attains a high performance at output stride = 8 during evaluation.</li>
  <li>DeepLabv3+: We extend DeepLabv3 to include a simple yet effective decoder module to refine the segmentation results especially along object boundaries. Furthermore, in this encoder-decoder structure one can arbitrarily control the resolution of extracted encoder features by atrous convolution to trade-off precision and runtime.</li>
</ol>

## Example output
The model will create a mask over the target objects with high accuracy.
<img src="images/segmentation.gif" />

## Read more about this
<ul>
  <li>Blog post: <a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">Semantic Image Segmentation with DeepLab in TensorFlow</a></li>
  <li><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">Blog post: TensorFlow Lite Now Faster with Mobile GPUs (Developer Preview)</a></li>
  <li><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab GitHub: DeepLab: Deep Labelling for Semantic Image Segmentation</a></li>
</ul>
