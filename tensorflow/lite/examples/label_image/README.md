# TensorFlow Lite C++ image classification demo

This example shows how you can load a pre-trained and converted
TensorFlow Lite model and use it to recognize objects in images.

Before you begin,
make sure you [have TensorFlow installed](https://www.tensorflow.org/install).

You also need to [install Bazel 26.1](https://docs.bazel.build/versions/master/install.html)
in order to build this example code. And be sure you have the Python `future`
module installed:

```
pip install future --user
```

## Build the example

First run `$TENSORFLOW_ROOT/configure`. To build for Android, set
Android NDK or configure NDK setting in
`$TENSORFLOW_ROOT/WORKSPACE` first.

Build it for desktop machines (tested on Ubuntu and OS X):

```
bazel build --cxxopt=-std=c++11 //tensorflow/lite/examples/label_image:label_image
```

Build it for Android ARMv8:

```
bazel build --config monolithic --cxxopt=-std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=arm64-v8a \
  //tensorflow/lite/examples/label_image:label_image
```

or

```
bazel build --config android_arm64 --config monolithic --cxxopt=-std=c++11 \
  //tensorflow/lite/examples/label_image:label_image
```

Build it for Android arm-v7a:

```
bazel build --config monolithic --cxxopt=-std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=armeabi-v7a \
  //tensorflow/lite/examples/label_image:label_image
```

or

```
bazel build --config android_arm --config monolithic --cxxopt=-std=c++11 \
  //tensorflow/lite/examples/label_image:label_image
```


## Download sample model and image

You can use any compatible model, but the following MobileNet v1 model offers
a good demonstration of a model trained to recognize 1,000 different objects.

```
# Get model
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp

# Get labels
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt

mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```

## Run the sample

```
bazel-bin/tensorflow/lite/examples/label_image/label_image \
  --tflite_model /tmp/mobilenet_v1_1.0_224.tflite \
  --labels /tmp/labels.txt \
  --image testdata/grace_hopper.bmp
```

You should see results like this:

```
Loaded model /tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
invoked
average time: 68.12 ms
0.860174: 653 653:military uniform
0.0481017: 907 907:Windsor tie
0.00786704: 466 466:bulletproof vest
0.00644932: 514 514:cornet, horn, trumpet, trump
0.00608029: 543 543:drumstick
```

See the `label_image.cc` source code for other command line options.
