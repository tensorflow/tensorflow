# Tensorflow Go Image Recognition Demo

This example shows how you can load a pre-trained TensorFlow network and use it
to recognize objects in images.

## Description

This demo uses a Google Inception model to classify image files that are passed
in on the command line.

## To build/run

The TensorFlow `GraphDef` that contains the model definition and weights
is not packaged in the repo because of its size. Instead, you must
first download the file to the `data` directory in the source tree:

```bash
$ mkdir tensorflow/examples/label_image_go/data/

$ wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip -O tensorflow/examples/label_image_go/data/inception_dec_2015.zip

$ unzip tensorflow/examples/label_image_go/data/inception_dec_2015.zip -d tensorflow/examples/label_image_go/data/
```

Then, as long as you've managed to build the main TensorFlow framework, you
should have everything you need to run this example installed already.

If you still don't have the Go libraries installed you can execute:

```bash
$ go generate github.com/tensorflow/tensorflow/tensorflow/contrib/go

$ go get github.com/tensorflow/tensorflow/tensorflow/contrib/go
```

Once extracted, see the labels file in the data directory for the possible
classifications, which are the 1,000 categories used in the Imagenet
competition.

To build it, run this commands:

```bash
$ go build -o label_image tensorflow/examples/label_image_go/main.go
```

That should build a binary executable that you can then run like this:

```bash
$ ./label_image tensorflow/examples/label_image/data/grace_hopper.jpg
```

This uses the example image on `tensorflow/examples/label_image/data/grace_hopper.jpg`,
and should output something similar to this:

```
military uniform : 0.6472986
suit : 0.04771947
academic gown : 0.02324065
bow tie : 0.015735544
bolo tie : 0.01450234
```
In this case, we're using the default image of [Admiral Grace
Hopper](https://en.wikipedia.org/wiki/Grace_Hopper), and you can see the
network correctly identifies she's wearing a military uniform, with a high
score of 0.6.

Next, try it out on your own images

For a more detailed look at this code, you can check out the Go section of the
[Inception tutorial](https://tensorflow.org/tutorials/image_recognition/).
