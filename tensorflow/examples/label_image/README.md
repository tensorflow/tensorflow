# Tensorflow C++ Image Recognition Demo

This example shows how you can load a pre-trained TensorFlow network and use it
to recognize objects in images.

## Description

This demo uses a Google Inception model to classify image files that are passed
in on the command line. See
[`googlenet_labels.txt`](data/googlenet_labels.txt)
for the possible classifications, which are the 1,000 categories used in the
Imagenet competition.

## To build/install/run

As long as you've managed to build the main TensorFlow framework, you should
have everything you need to run this example installed already.

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
I tensorflow/examples/label_image/main.cc:200] military uniform (866): 0.902268
I tensorflow/examples/label_image/main.cc:200] bow tie (817): 0.05407
I tensorflow/examples/label_image/main.cc:200] suit (794): 0.0113195
I tensorflow/examples/label_image/main.cc:200] bulletproof vest (833): 0.0100269
I tensorflow/examples/label_image/main.cc:200] bearskin (849): 0.00649746
```
In this case, we're using the default image of Admiral Grace Hopper, and you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.9.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image --image=my_image.png
```
