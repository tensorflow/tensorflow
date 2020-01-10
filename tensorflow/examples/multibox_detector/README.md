# TensorFlow C++ MultiBox Object Detection Demo

This example shows how you can load a pre-trained TensorFlow network and use it
to detect objects in images in C++. For an alternate implementation see the
[Android TensorFlow demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)

## Description

This demo uses a model based on [Scalable Object Detection using Deep NeuralNetworks](https://arxiv.org/abs/1312.2249) to detect people in images passed in from
the command line. This is the same model also used in the Android TensorFlow
demo for real-time person detection and tracking in the camera preview.

## To build/install/run

The TensorFlow `GraphDef` that contains the model definition and weights is not
packaged in the repo because of its size. Instead, you must first download the
file to the `data` directory in the source tree:

```bash
$ wget https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip -O tensorflow/examples/multibox_detector/data/mobile_multibox_v1a.zip

$ unzip tensorflow/examples/multibox_detector/data/mobile_multibox_v1a.zip -d tensorflow/examples/multibox_detector/data/
```

Then, as long as you've managed to build the main TensorFlow framework, you
should have everything you need to run this example installed already.

Once extracted, see the box priors file in the data directory. This file
contains means and standard deviations for all 784 possible detections,
normalized from 0-1 in left top right bottom order.

To build it, run this command:

```bash
$ bazel build --config opt tensorflow/examples/multibox_detector/...
```

That should build a binary executable that you can then run like this:

```bash
$ bazel-bin/tensorflow/examples/multibox_detector/detect_objects --image_out=$HOME/x20/surfers_labeled.png
```

This uses the default example image that ships with the framework, and should
output something similar to this:

```
I0125 18:24:13.804047    8677 main.cc:293] ===== Top 5 Detections ======
I0125 18:24:13.804058    8677 main.cc:307] Detection 0: L:324.542 T:76.5764 R:373.26 B:214.957 (635) score: 0.267425
I0125 18:24:13.804077    8677 main.cc:307] Detection 1: L:332.896 T:76.2751 R:372.116 B:204.614 (523) score: 0.245334
I0125 18:24:13.804087    8677 main.cc:307] Detection 2: L:306.605 T:76.2228 R:371.356 B:217.32 (634) score: 0.216121
I0125 18:24:13.804096    8677 main.cc:307] Detection 3: L:143.918 T:86.0909 R:187.333 B:195.885 (387) score: 0.171368
I0125 18:24:13.804104    8677 main.cc:307] Detection 4: L:144.915 T:86.2675 R:185.243 B:165.246 (219) score: 0.169244
```

In this case, we're using a public domain stock image of surfers walking on the
beach, and the top two few detections are of the two on the right. Adding more
detections with --num_detections=N will also include the surfer on the left,
and eventually non-person boxes below a certain threshold.

You can visually inspect the detections by viewing the resulting png file
'~/surfers_labeled.png'.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ bazel-bin/tensorflow/examples/multibox_detector/detect_objects --image=my_image.png
```

For another implementation of this work, you can check out the [Android
TensorFlow demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).
