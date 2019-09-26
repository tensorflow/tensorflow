# Hair segmentation

## Get started

Hair segmentation model produces a high-quality hair segmentation mask that is
well suited for AR effects, e.g. virtual hair recoloring.

For a working demo of a live hair recoloring Android app using the model, check
out this example by [MediaPipe](https://mediapipe.readthedocs.io/en/latest/):

<a class="button button-primary" href="https://github.com/google/mediapipe/blob/master/mediapipe/docs/hair_segmentation_mobile_gpu.md">Android
example</a>
<a class="button button-primary" href="https://github.com/google/mediapipe/raw/master/mediapipe/models/hair_segmentation.tflite">Download
starter model</a>

### How it works

Hair segmentation refers to computer vision techniques that detect human hair in
images and videos. To be clear, this technology is not recognizing who is in an
image. The algorithm only estimates where is hair on an image and where is
everything else.

The model takes a video frame as input and returns a mask that tests if a pixel
is a hair. For better results, this resulting mask is used as an additional
input to the next frame.

### Model architecture

Standard hourglass segmentation network architecture with skip connections used
for this model. The input is a 512x512x4 matrix. Channels are red, green, blue,
previous mask or zeros for the first frame. The output is a 512x512x2 matrix
with a background in the first channel and a hair mask in the second.

![Model architecture](https://storage.googleapis.com/download.tensorflow.org/models/tflite/hair_segmentation/model_architecture.png)

### Examples of hair recoloring

![Sample](https://storage.googleapis.com/download.tensorflow.org/models/tflite/hair_segmentation/demo.gif)

### Read more about hair segmentation

*   [Paper: Real-time Hair segmentation and recoloring on Mobile GPUs](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation)
