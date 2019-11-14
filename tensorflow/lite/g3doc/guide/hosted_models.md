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

## Question and Answer

For more information about text classification with Mobile BERT, see
<a href="../bert_qa/overview.md">Question And Answer</a>.

### Quantized models

<a href="../performance/post_training_quantization">Quantized</a> image
classification models offer the smallest model size and fastest performance, at
the expense of accuracy. The performance values are measured on Pixel 3 on
Android 10.

Model name                  | Paper and model                                                                                                                                                                   | Model size | Top-1 accuracy | Top-5 accuracy | CPU, 4 threads | NNAPI
--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------: | -------------: | -------------: | -------------: | ----:
Mobilenet_V1_0.25_128_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz) | 0.5 Mb     | 39.5%          | 64.4%          | 0.8 ms         | 2 ms
Mobilenet_V1_0.25_160_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) | 0.5 Mb     | 42.8%          | 68.1%          | 1.3 ms         | 2.4 ms
Mobilenet_V1_0.25_192_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz) | 0.5 Mb     | 45.7%          | 70.8%          | 1.8 ms         | 2.6 ms
Mobilenet_V1_0.25_224_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz) | 0.5 Mb     | 48.2%          | 72.8%          | 2.3 ms         | 2.9 ms
Mobilenet_V1_0.50_128_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz)  | 1.4 Mb     | 54.9%          | 78.1%          | 1.7 ms         | 2.6 ms
Mobilenet_V1_0.50_160_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz)  | 1.4 Mb     | 57.2%          | 80.5%          | 2.6 ms         | 2.9 ms
Mobilenet_V1_0.50_192_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz)  | 1.4 Mb     | 59.9%          | 82.1%          | 3.6 ms         | 3.3 ms
Mobilenet_V1_0.50_224_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz)  | 1.4 Mb     | 61.2%          | 83.2%          | 4.7 ms         | 3.6 ms
Mobilenet_V1_0.75_128_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz) | 2.6 Mb     | 55.9%          | 79.1%          | 3.1 ms         | 3.2 ms
Mobilenet_V1_0.75_160_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz) | 2.6 Mb     | 62.4%          | 83.7%          | 4.7 ms         | 3.8 ms
Mobilenet_V1_0.75_192_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz) | 2.6 Mb     | 66.1%          | 86.2%          | 6.4 ms         | 4.2 ms
Mobilenet_V1_0.75_224_quant | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz) | 2.6 Mb     | 66.9%          | 86.9%          | 8.5 ms         | 4.8 ms
Mobilenet_V1_1.0_128_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz)  | 4.3 Mb     | 63.3%          | 84.1%          | 4.8 ms         | 3.8 ms
Mobilenet_V1_1.0_160_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz)  | 4.3 Mb     | 66.9%          | 86.7%          | 7.3 ms         | 4.6 ms
Mobilenet_V1_1.0_192_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz)  | 4.3 Mb     | 69.1%          | 88.1%          | 9.9 ms         | 5.2 ms
Mobilenet_V1_1.0_224_quant  | [paper](https://arxiv.org/pdf/1712.05877.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)  | 4.3 Mb     | 70.0%          | 89.0%          | 13 ms          | 6.0 ms
Mobilenet_V2_1.0_224_quant  | [paper](https://arxiv.org/abs/1806.08342), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz)              | 3.4 Mb     | 70.8%          | 89.9%          | 12 ms          | 6.9 ms
Inception_V1_quant          | [paper](https://arxiv.org/abs/1409.4842), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz)                          | 6.4 Mb     | 70.1%          | 89.8%          | 39 ms          | 36 ms
Inception_V2_quant          | [paper](https://arxiv.org/abs/1512.00567), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz)                         | 11 Mb      | 73.5%          | 91.4%          | 59 ms          | 18 ms
Inception_V3_quant          | [paper](https://arxiv.org/abs/1806.08342),[tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz)                       | 23 Mb      | 77.5%          | 93.7%          | 148 ms         | 74 ms
Inception_V4_quant          | [paper](https://arxiv.org/abs/1602.07261), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz)                         | 41 Mb      | 79.5%          | 93.9%          | 268 ms         | 155 ms

Note: The model files include both TF Lite FlatBuffer and Tensorflow frozen
Graph.

Note: Performance numbers were benchmarked on Pixel-2 using single thread large
core. Accuracy numbers were computed using the
[TFLite accuracy tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/accuracy/ilsvrc).

### Floating point models

Floating point models offer the best accuracy, at the expense of model size and
performance. <a href="../performance/gpu">GPU acceleration</a> requires the use
of floating point models. The performance values are measured on Pixel 3 on
Android 10.

Model name            | Paper and model                                                                                                                                                                           | Model size | Top-1 accuracy | Top-5 accuracy | CPU, 4 threads | GPU    | NNAPI
--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------: | -------------: | -------------: | -------------: | -----: | ----:
DenseNet              | [paper](https://arxiv.org/abs/1608.06993), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz)            | 43.6 Mb    | 64.2%          | 85.6%          | 195 ms         | 60 ms  | 1656 ms
SqueezeNet            | [paper](https://arxiv.org/abs/1602.07360), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz)          | 5.0 Mb     | 49.0%          | 72.9%          | 36 ms          | 9.5 ms | 18.5 ms
NASNet mobile         | [paper](https://arxiv.org/abs/1707.07012), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz)       | 21.4 Mb    | 73.9%          | 91.5%          | 56 ms          | ---    | 102 ms
NASNet large          | [paper](https://arxiv.org/abs/1707.07012), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz)        | 355.3 Mb   | 82.6%          | 96.1%          | 1170 ms        | ---    | 648 ms
ResNet_V2_101         | [paper](https://arxiv.org/abs/1603.05027), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz)                                   | 178.3 Mb   | 76.8%          | 93.6%          | 526 ms         | 92 ms  | 1572 ms
Inception_V3          | [paper](http://arxiv.org/abs/1512.00567), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz)         | 95.3 Mb    | 77.9%          | 93.8%          | 249 ms         | 56 ms  | 148 ms
Inception_V4          | [paper](http://arxiv.org/abs/1602.07261), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz)         | 170.7 Mb   | 80.1%          | 95.1%          | 486 ms         | 93 ms  | 291 ms
Inception_ResNet_V2   | [paper](https://arxiv.org/abs/1602.07261), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) | 121.0 Mb   | 77.5%          | 94.0%          | 422 ms         | 100 ms | 201 ms
Mobilenet_V1_0.25_128 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz)               | 1.9 Mb     | 41.4%          | 66.2%          | 1.2 ms         | 1.6 ms | 3 ms
Mobilenet_V1_0.25_160 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_160.tgz)               | 1.9 Mb     | 45.4%          | 70.2%          | 1.7 ms         | 1.7 ms | 3.2 ms
Mobilenet_V1_0.25_192 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_192.tgz)               | 1.9 Mb     | 47.1%          | 72.0%          | 2.4 ms         | 1.8 ms | 3.0 ms
Mobilenet_V1_0.25_224 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz)               | 1.9 Mb     | 49.7%          | 74.1%          | 3.3 ms         | 1.8 ms | 3.6 ms
Mobilenet_V1_0.50_128 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz)                | 5.3 Mb     | 56.2%          | 79.3%          | 3.0 ms         | 1.7 ms | 3.2 ms
Mobilenet_V1_0.50_160 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz)                | 5.3 Mb     | 59.0%          | 81.8%          | 4.4 ms         | 2.0 ms | 4.0 ms
Mobilenet_V1_0.50_192 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_192.tgz)                | 5.3 Mb     | 61.7%          | 83.5%          | 6.0 ms         | 2.5 ms | 4.8 ms
Mobilenet_V1_0.50_224 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz)                | 5.3 Mb     | 63.2%          | 84.9%          | 7.9 ms         | 2.8 ms | 6.1 ms
Mobilenet_V1_0.75_128 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_128.tgz)               | 10.3 Mb    | 62.0%          | 83.8%          | 5.5 ms         | 2.6 ms | 5.1 ms
Mobilenet_V1_0.75_160 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_160.tgz)               | 10.3 Mb    | 65.2%          | 85.9%          | 8.2 ms         | 3.1 ms | 6.3 ms
Mobilenet_V1_0.75_192 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_192.tgz)               | 10.3 Mb    | 67.1%          | 87.2%          | 11.0 ms        | 4.5 ms | 7.2 ms
Mobilenet_V1_0.75_224 | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz)               | 10.3 Mb    | 68.3%          | 88.1%          | 14.6 ms        | 4.9 ms | 9.9 ms
Mobilenet_V1_1.0_128  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz)                | 16.9 Mb    | 65.2%          | 85.7%          | 9.0 ms         | 4.4 ms | 6.3 ms
Mobilenet_V1_1.0_160  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_160.tgz)                | 16.9 Mb    | 68.0%          | 87.7%          | 13.4 ms        | 5.0 ms | 8.4 ms
Mobilenet_V1_1.0_192  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_192.tgz)                | 16.9 Mb    | 69.9%          | 89.1%          | 18.1 ms        | 6.3 ms | 10.6 ms
Mobilenet_V1_1.0_224  | [paper](https://arxiv.org/pdf/1704.04861.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)                | 16.9 Mb    | 71.0%          | 89.9%          | 24.0 ms        | 6.5 ms | 13.8 ms
Mobilenet_V2_1.0_224  | [paper](https://arxiv.org/pdf/1801.04381.pdf), [tflite&pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)                        | 14.0 Mb    | 71.8%          | 90.6%          | 17.5 ms        | 6.2 ms | 11.23 ms

### AutoML mobile models

The following image classification models were created using
<a href="https://cloud.google.com/automl/">Cloud AutoML</a>. The performance
values are measured on Pixel 3 on Android 10.

Model Name       | Paper and model                                                                                                                                                | Model size | Top-1 accuracy | Top-5 accuracy | CPU, 4 threads | GPU     | NNAPI
---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------: | -------------: | -------------: | -------------: | ------: | ----:
MnasNet_0.50_224 | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.5_224_09_07_2018.tgz)  | 8.5 Mb     | 68.03%         | 87.79%         | 9.5 ms         | 5.9 ms  | 16.6 ms
MnasNet_0.75_224 | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.75_224_09_07_2018.tgz) | 12 Mb      | 71.72%         | 90.17%         | 13.7 ms        | 7.1 ms  | 16.7 ms
MnasNet_1.0_96   | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_96_09_07_2018.tgz)   | 17 Mb      | 62.33%         | 83.98%         | 5.6 ms         | 5.4 ms  | 12.1 ms
MnasNet_1.0_128  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_128_09_07_2018.tgz)  | 17 Mb      | 67.32%         | 87.70%         | 7.5 ms         | 5.8 ms  | 12.9 ms
MnasNet_1.0_160  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_160_09_07_2018.tgz)  | 17 Mb      | 70.63%         | 89.58%         | 11.1 ms        | 6.7 ms  | 14.2 ms
MnasNet_1.0_192  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_192_09_07_2018.tgz)  | 17 Mb      | 72.56%         | 90.76%         | 14.5 ms        | 7.7 ms  | 16.6 ms
MnasNet_1.0_224  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_224_09_07_2018.tgz)  | 17 Mb      | 74.08%         | 91.75%         | 19.4 ms        | 8.7 ms  | 19 ms
MnasNet_1.3_224  | [paper](https://arxiv.org/abs/1807.11626), [tflite&pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.3_224_09_07_2018.tgz)  | 24 Mb      | 75.24%         | 92.55%         | 27.9 ms        | 10.6 ms | 22.0 ms

Note: Performance numbers were benchmarked on Pixel-1 using single thread large
BIG core.

## Object detection

For more information about object detection, see
<a href="../models/object_detection/overview.md">Object detection</a>.

The object detection model we currently host is
**coco_ssd_mobilenet_v1_1.0_quant_2018_06_29**.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip">Download
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
