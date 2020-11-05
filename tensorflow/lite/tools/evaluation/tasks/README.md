# TFLite Model Task Evaluation

This page describes how you can check the accuracy of quantized models to verify that any degradation in accuracy is within acceptable limits.

## Tools
There are three different binaries which are supported. A brief description of each is provided below.

### [Inference Diff Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff#inference-diff-tool)
This binary compares TensorFlow Lite execution in single-threaded CPU inference and user-defined inference.



### [Image Classification Evaluation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification#image-classification-evaluation-based-on-ilsvrc-2012-task)
This binary evaluates TensorFlow Lite models trained for the [ILSVRC 2012 image classification task.](http://www.image-net.org/challenges/LSVRC/2012/)

### [Object Detection Evaluation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection#object-detection-evaluation-using-the-2014-coco-minival-dataset)
This binary evaluates TensorFlow Lite models trained for the bounding box-based [COCO Object Detection](https://cocodataset.org/#detection-eval) task.