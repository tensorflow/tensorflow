# Face detection

## Get started

BlazeFace is a lightweight face detection model, designed specifically for
selfie use-case for mobile devices. It works for faces up to 2 meters from
camera and provides 6 additional facial keypoints, which allows to estimate face
rotation angles and do basic AR-effects on top of it.

For a working demo of an ultrafast realtime face detection Android app using the
model, check out this example by
[MediaPipe](https://mediapipe.readthedocs.io/en/latest/):

<a class="button button-primary" href="https://github.com/google/mediapipe/blob/master/mediapipe/docs/face_detection_mobile_gpu.md">Android
example</a>
<a class="button button-primary" href="https://github.com/google/mediapipe/raw/master/mediapipe/models/face_detection_front.tflite">Download
starter model</a>

### How it works

BlazeFace is a lightweight and well-performing face detector tailored for mobile
GPU inference. It runs at a speed of 200–1000+ FPS on flagship devices. This
super-realtime performance enables it to be applied to any augmented reality
pipeline that requires an accurate facial region of interest as an input for
task-specific models, such as 2D/3D facial keypoint or geometry estimation,
facial features or expression classification, and face region segmentation.

The new techniques implemented in the model are:

*   lightweight feature extraction network
*   GPU-friendly anchor scheme
*   improved tie resolution strategy alternative to non-maximum suppression

Per each prediction, BlazeFace predicts face bounding box and 2D coordinates for
6 facial keypoints:

Id  | Part
--- | -----------------
0   | left_eye
1   | right_eye
2   | nose_tip
3   | mouth_center
4   | left_ear_tragion
5   | right_ear_tragion

### Examples of face detection

![Demo](https://storage.googleapis.com/download.tensorflow.org/models/tflite/face_detection/demo.gif)

### How it performs

Model works in several predefined recommended resolutions, depending on input
screen aspect ratio: **128x96**, **128x128**, **96x128**. For resolution
**128x96** inference times shown below:

Device                         | Inference time (ms)
------------------------------ | -------------------
Apple iPhone 7                 | 1.8
Apple iPhone XS                | 0.6
Google Pixel 3                 | 3.4
Huawei P20                     | 5.8
Samsung Galaxy S9+ (SM-G965U1) | 3.7

### Read more about BlazeFace

*   [Paper: BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://sites.google.com/corp/view/perception-cv4arvr/blazeface)
