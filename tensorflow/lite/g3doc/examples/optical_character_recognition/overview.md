# Optical character recognition (OCR)

Optical character recognition (OCR) is the process of recognizing characters
from images using computer vision and machine learning techniques. This
reference app demos how to use TensorFlow Lite to do OCR. It uses a combination
of
[text detection model](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1)
and a
[text recognition model](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)
as an OCR pipeline to recognize text characters.

## Get started

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend
exploring the following example application that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/optical_character_recognition/android">Android
example</a>

If you are using a platform other than Android, or you are already familiar with
the [TensorFlow Lite APIs](https://www.tensorflow.org/api_docs/python/tf/lite),
you can download the models from [TF Hub](https://tfhub.dev/).

## How it works

OCR tasks are often broken down into 2 stages. First, we use a text detection
model to detect the bounding boxes around possible texts. Second, we feed
processed bounding boxes into a text recognition model to determine specific
characters inside the bounding boxes (we also need to do Non-Maximal Supression,
perspective transformation and etc. beforing text recoginition). In our case,
both models are from TensorFlow Hub and they are FP16 quantized models.

## Performance benchmarks

Performance benchmark numbers are generated with the tool described
[here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>Device </th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>
      <a href="https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1">Text Detection</a>
    </td>
    <td>45.9 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>181.93ms*</td>
     <td>89.77ms*</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2">Text Recognition</a>
    </td>
    <td>16.8 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>338.33ms*</td>
     <td>N/A**</td>
  </tr>
</table>

\* 4 threads used.

\** this model could not use GPU delegate since we need TensorFlow ops to run it

## Inputs

The text detection model accepts a 4-D `float32` Tensor of (1, 320, 320, 3) as
input.

The text recognition model accepts a 4-D `float32` Tensor of (1, 31, 200, 1) as
input.

## Outputs

The text detection model returns a 4-D `float32` Tensor of shape (1, 80, 80, 5)
as bounding box and a 4-D `float32` Tensor of shape (1,80, 80, 5) as detection
score.

The text recognition model returns a 2-D `float32` Tensor of shape (1, 48) as
the mapping indices to the alphabet list '0123456789abcdefghijklmnopqrstuvwxyz'

## Limitations

*   The current
    [text recognition model](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)
    is trained using synthetic data with English letters and numbers, so only
    English is supported.

*   The models are not general enough for OCR in the wild (say, random images
    taken by a smartphone camera in a low lighting condition).

So we have chosen 3 Google product logos only to demonstrate how to do OCR with
TensorFlow Lite. If you are looking for a ready-to-use production-grade OCR
product, you should consider
[Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition).
ML Kit, which uses TFLite underneath, should be sufficient for most OCR use
cases, but there are some cases where you may want to build your own OCR
solution with TFLite. Some examples are:

*   You have your own text detection/recognition TFLite models that you would
    like to use
*   You have special business requirements (i.e., recognizing texts that are
    upside down) and need to customize the OCR pipeline
*   You want to support languages not covered by ML Kit
*   Your target user devices don’t necessarily have Google Play services
    installed

## References

*   OpenCV text detection/recognition example:
    https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
*   OCR TFLite community project by community contributors:
    https://github.com/tulasiram58827/ocr_tflite
*   OpenCV text detection:
    https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
*   Deep Learning based Text Detection Using OpenCV:
    https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
