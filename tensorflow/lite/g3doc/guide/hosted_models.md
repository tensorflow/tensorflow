# Hosted models

The following is an incomplete list of pre-trained models optimized to work with
TensorFlow Lite.

To get started choosing a model, visit <a href="../models">Models</a>.

Note: The best model for a given application depends on your requirements. For
example, some applications might benefit from higher accuracy, while others
require a small model size. You should test your application with a variety of
models to find the optimal balance between size, performance, and accuracy.

## Image classification

For more information about image classification, see
<a href="../image_classification/overview.md">Image classification</a>.

### Quantized models

<a href="../performance/post_training_quantization.md">Quantized</a> image
classification models offer the smallest model size and fastest performance, at
the expense of accuracy.

Model name                  | Paper and model                                                                                                                                           | Model size | Top-1 accuracy | Top-5 accuracy | TF Lite performance
--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------: | -------------: | -------------: | ------------------:
Mobilenet_V1_0.25_128_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz) | 0.5 Mb     | 39.5%          | 64.4%          | 3.7 ms
Mobilenet_V1_0.25_160_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) | 0.5 Mb     | 42.8%          | 68.1%          | 5.5 ms
Mobilenet_V1_0.25_192_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz) | 0.5 Mb     | 45.7%          | 70.8%          | 7.9 ms
Mobilenet_V1_0.25_224_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz) | 0.5 Mb     | 48.2%          | 72.8%          | 10.4 ms
Mobilenet_V1_0.50_128_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz)  | 1.4 Mb     | 54.9%          | 78.1%          | 8.8 ms
Mobilenet_V1_0.50_160_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz)  | 1.4 Mb     | 57.2%          | 80.5%          | 13.0 ms
Mobilenet_V1_0.50_192_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz)  | 1.4 Mb     | 59.9%          | 82.1%          | 18.3 ms
Mobilenet_V1_0.50_224_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz)  | 1.4 Mb     | 61.2%          | 83.2%          | 24.7 ms
Mobilenet_V1_0.75_128_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz) | 2.6 Mb     | 55.9%          | 79.1%          | 16.2 ms
Mobilenet_V1_0.75_160_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz) | 2.6 Mb     | 62.4%          | 83.7%          | 24.3 ms
Mobilenet_V1_0.75_192_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz) | 2.6 Mb     | 66.1%          | 86.2%          | 33.8 ms
Mobilenet_V1_0.75_224_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz) | 2.6 Mb     | 66.9%          | 86.9%          | 45.4 ms
Mobilenet_V1_1.0_128_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz)  | 4.3 Mb     | 63.3%          | 84.1%          | 24.9 ms
Mobilenet_V1_1.0_160_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz)  | 4.3 Mb     | 66.9%          | 86.7%          | 37.4 ms
Mobilenet_V1_1.0_192_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz)  | 4.3 Mb     | 69.1%          | 88.1%          | 51.9 ms
Mobilenet_V1_1.0_224_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)  | 4.3 Mb     | 70.0%          | 89.0%          | 70.2 ms
Mobilenet_V2_1.0_224_quant  | [paper](https://arxiv.org/abs/1806.08342), [tflite&pb](http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz)              | 3.4 Mb     | 70.8%          | 89.9%          | 80.3 ms
Inception_V1_quant          | [paper](https://arxiv.org/abs/1409.4842), [tflite&pb](http://download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz)                          | 6.4 Mb     | 70.1%          | 89.8%          | 154.5 ms
Inception_V2_quant          | [paper](https://arxiv.org/abs/1512.00567), [tflite&pb](http://download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz)                         | 11 Mb      | 73.5%          | 91.4%          | 235.0 ms
Inception_V3_quant          | [paper](https://arxiv.org/abs/1806.08342),[tflite&pb](http://download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz)                       | 23 Mb      | 77.5%          | 93.7%          | 637 ms
Inception_V4_quant          | [paper](https://arxiv.org/abs/1602.07261), [tflite&pb](http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz)                         | 41 Mb      | 79.5%          | 93.9%          | 1250.8 ms

Note: The model files include both TF Lite FlatBuffer and Tensorflow frozen
Graph.

Note: Performance numbers were benchmarked on Pixel-2 using single thread large
core. Accuracy numbers were computed using the
[TFLite accuracy tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/accuracy/ilsvrc).

### Floating point models

Floating point models offer the best accuracy, at the expense of model size and
performance. <a href="../performance/gpu.md">GPU acceleration</a> requires the
use of floating point models.

Model name            | Paper and model                                                                                                                                                                           | Model size | Top-1 accuracy | Top-5 accuracy | TF Lite performance | Tensorflow performance
--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------: | -------------: | -------------: | ------------------: | ---------------------:
DenseNet              | [paper](https://arxiv.org/abs/1608.06993), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz)            | 43.6 Mb    | 64.2%          | 85.6%          | 894 ms              | 1262 ms
SqueezeNet            | [paper](https://arxiv.org/abs/1602.07360), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz)          | 5.0 Mb     | 49.0%          | 72.9%          | 224 ms              | 255 ms
NASNet mobile         | [paper](https://arxiv.org/abs/1707.07012), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz)       | 21.4 Mb    | 73.9%          | 91.5%          | 261 ms              | 389 ms
NASNet large          | [paper](https://arxiv.org/abs/1707.07012), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz)        | 355.3 Mb   | 82.6%          | 96.1%          | 6697 ms             | 7940 ms
ResNet_V2_101         | [paper](https://arxiv.org/abs/1603.05027), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz)                                   | 178.3 Mb   | 76.8%          | 93.6%          | 1880 ms             | 1970 ms
Inception_V3          | [paper](http://arxiv.org/abs/1512.00567), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz)         | 95.3 Mb    | 77.9%          | 93.8%          | 1433 ms             | 1522 ms
Inception_V4          | [paper](http://arxiv.org/abs/1602.07261), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz)         | 170.7 Mb   | 80.1%          | 95.1%          | 2986 ms             | 3139 ms
Inception_ResNet_V2   | [paper](https://arxiv.org/abs/1602.07261), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) | 121.0 Mb   | 77.5%          | 94.0%          | 2731 ms             | 2926 ms
Mobilenet_V1_0.25_128 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz)                                       | 1.9 Mb     | 41.4%          | 66.2%          | 6.2 ms              | 13.0 ms
Mobilenet_V1_0.25_160 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_160.tgz)                                       | 1.9 Mb     | 45.4%          | 70.2%          | 8.6 ms              | 19.5 ms
Mobilenet_V1_0.25_192 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_192.tgz)                                       | 1.9 Mb     | 47.1%          | 72.0%          | 12.1 ms             | 27.8 ms
Mobilenet_V1_0.25_224 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz)                                       | 1.9 Mb     | 49.7%          | 74.1%          | 16.2 ms             | 37.3 ms
Mobilenet_V1_0.50_128 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz)                                        | 5.3 Mb     | 56.2%          | 79.3%          | 18.1 ms             | 29.9 ms
Mobilenet_V1_0.50_160 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz)                                        | 5.3 Mb     | 59.0%          | 81.8%          | 26.8 ms             | 45.9 ms
Mobilenet_V1_0.50_192 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_192.tgz)                                        | 5.3 Mb     | 61.7%          | 83.5%          | 35.6 ms             | 65.3 ms
Mobilenet_V1_0.50_224 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz)                                        | 5.3 Mb     | 63.2%          | 84.9%          | 47.6 ms             | 164.2 ms
Mobilenet_V1_0.75_128 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_128.tgz)                                       | 10.3 Mb    | 62.0%          | 83.8%          | 34.6 ms             | 48.7 ms
Mobilenet_V1_0.75_160 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_160.tgz)                                       | 10.3 Mb    | 65.2%          | 85.9%          | 51.3 ms             | 75.2 ms
Mobilenet_V1_0.75_192 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_192.tgz)                                       | 10.3 Mb    | 67.1%          | 87.2%          | 71.7 ms             | 107.0 ms
Mobilenet_V1_0.75_224 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz)                                       | 10.3 Mb    | 68.3%          | 88.1%          | 95.7 ms             | 143.4 ms
Mobilenet_V1_1.0_128  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz)                                        | 16.9 Mb    | 65.2%          | 85.7%          | 57.4 ms             | 76.8 ms
Mobilenet_V1_1.0_160  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_160.tgz)                                        | 16.9 Mb    | 68.0%          | 87.7%          | 86.0 ms             | 117.7 ms
Mobilenet_V1_1.0_192  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_192.tgz)                                        | 16.9 Mb    | 69.9%          | 89.1%          | 118.6 ms            | 167.3 ms
Mobilenet_V1_1.0_224  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)                                        | 16.9 Mb    | 71.0%          | 89.9%          | 160.1 ms            | 224.3 ms
Mobilenet_V2_1.0_224  | [paper](https://arxiv.org/pdf/1801.04381.pdf), [tflite&pb](http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)                                                | 14.0 Mb    | 71.8%          | 90.6%          | 117 ms              |

### AutoML mobile models

The following image classification models were created using
<a href="https://cloud.google.com/automl/">Cloud AutoML</a>.

Model Name       | Paper and model                                                                                                                                                | Model size | Top-1 accuracy | Top-5 accuracy | TF Lite performance
---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------: | -------------: | -------------: | ------------------:
MnasNet_0.50_224 | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.5_224_09_07_2018.tgz)  | 8.5 Mb     | 68.03%         | 87.79%         | 37 ms
MnasNet_0.75_224 | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.75_224_09_07_2018.tgz) | 12 Mb      | 71.72%         | 90.17%         | 61 ms
MnasNet_1.0_96   | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_96_09_07_2018.tgz)   | 17 Mb      | 62.33%         | 83.98%         | 23 ms
MnasNet_1.0_128  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_128_09_07_2018.tgz)  | 17 Mb      | 67.32%         | 87.70%         | 34 ms
MnasNet_1.0_160  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_160_09_07_2018.tgz)  | 17 Mb      | 70.63%         | 89.58%         | 51 ms
MnasNet_1.0_192  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_192_09_07_2018.tgz)  | 17 Mb      | 72.56%         | 90.76%         | 70 ms
MnasNet_1.0_224  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_224_09_07_2018.tgz)  | 17 Mb      | 74.08%         | 91.75%         | 93 ms
MnasNet_1.3_224  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.3_224_09_07_2018.tgz)  | 24 Mb      | 75.24%         | 92.55%         | 152 ms

Note: Performance numbers were benchmarked on Pixel-1 using single thread large
BIG core.

## Object detection

For more information about object detection, see
<a href="../models/object_detection/overview.md">Object detection</a>.

The object detection model we currently host is
**coco_ssd_mobilenet_v1_1.0_quant_2018_06_29**.

<a class="button button-primary" href="http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip">Download
model and labels</a>

## Pose estimation

For more information about pose estimation, see
<a href="../models/pose_estimation/overview.md">Pose estimation</a>.

The pose estimation model we currently host is
**multi_person_mobilenet_v1_075_float**.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite">Download
model</a>

## Image segmentation

For more information about image segmentation, see
<a href="../models/segmentation/overview.md">Segmentation</a>.

The image segmentation model we currently host is **deeplabv3_257_mv_gpu**.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite">Download
model</a>

## Smart reply

For more information about smart reply, see
<a href="../models/smart_reply/overview.md">Smart reply</a>.

The smart reply model we currently host is **smartreply_1.0_2017_11_01**.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/smartreply_1.0_2017_11_01.zip">Download
model</a>
