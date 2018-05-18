# TensorFlow C++ and Python Image Recognition Demo

This example shows how you can load a pre-trained TensorFlow network and use it
to recognize objects in images in C++. For Java see the [Java
README](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java),
and for Go see the [godoc
example](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go#ex-package).

## Description

This demo uses a Google Inception model to classify image files that are passed
in on the command line.

## To build/install/run

The TensorFlow `GraphDef` that contains the model definition and weights is not
packaged in the repo because of its size. Instead, you must first download the
file to the `data` directory in the source tree:

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C tensorflow/examples/label_image/data -xz
```

Then, as long as you've managed to build the main TensorFlow framework, you
should have everything you need to run this example installed already.

Once extracted, see the labels file in the data directory for the possible
classifications, which are the 1,000 categories used in the Imagenet
competition.

To build it, run this command:

```bash
$ bazel build tensorflow/examples/label_image/...
```

That should build a binary executable that you can then run like this:

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image
```

This uses the default example image that ships with the framework, and should
output something similar to this:

```
I tensorflow/examples/label_image/main.cc:206] military uniform (653): 0.834306
I tensorflow/examples/label_image/main.cc:206] mortarboard (668): 0.0218692
I tensorflow/examples/label_image/main.cc:206] academic gown (401): 0.0103579
I tensorflow/examples/label_image/main.cc:206] pickelhaube (716): 0.00800814
I tensorflow/examples/label_image/main.cc:206] bulletproof vest (466): 0.00535088
```

In this case, we're using the default image of Admiral Grace Hopper, and you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.8.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image --image=my_image.png
```

For a more detailed look at this code, you can check out the C++ section of the
[Inception tutorial](https://www.tensorflow.org/tutorials/image_recognition/).

## Python implementation

label_image.py is a python implementation that provides code corresponding
to the C++ code here. This gives more intuitive mapping between C++ and
Python than the Python code mentioned in the
[Inception tutorial](https://www.tensorflow.org/tutorials/image_recognition/).
and could be easier to add visualization or debug code.


`bazel-bin/tensorflow/examples/label_image/label_image_py` should be there after
```bash
$ bazel build tensorflow/examples/label_image/...
```

Run

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image_py
```

Or, with tensorflow python package installed, you can run it like:
```bash
$ python3 tensorflow/examples/label_image/label_image.py
```

And get result similar to this:
```
military uniform 0.834305
mortarboard 0.0218694
academic gown 0.0103581
pickelhaube 0.00800818
bulletproof vest 0.0053509
```
