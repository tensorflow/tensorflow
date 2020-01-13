# MNIST classifcation Example

This example shows how you can use Tensorflow Lite micro to run a small 500
kilobyte fully connected MNIST classifier. It is intended to demonstrate the
minimal amount of code needed to export a model from a Tensorflow graph and
integrate
it into a Tensorflow lite micro. The classifier currently works with pre-loaded
test input used to validate the model, however if interfaced with a camera 
module then this example could work in a real world application.

## Table of Contents
-   [Getting Started](#getting-started)
-   [Training and Exporting the Model](#training-and-exporting-the-model)

## Getting Started

To compile and test this example on a desktop Linux or MacOS machine, download
[the TensorFlow source code](https://github.com/tensorflow/tensorflow), `cd`
into the source directory from a terminal, and then run the following command:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile
```

You only need to do this once, it will take a few minutes to download the
frameworks the code uses such as
[CMSIS](https://developer.arm.com/embedded/cmsis) and
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, you can build the MNIST demo with the command below:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile mnist_demo
```

This will have produced a native binary in the location:

```
tensorflow/lite/experimental/micro/tools/make/gen/<your platform>/bin
```

Executing this binary should produce the output below, testing and evaluating
the MNIST classifier on a set of compiled in test data:

```
Parsing MNIST classifier model FlatBuffer.
Model parsed.

Build interpreter.
Done.

Details of input tensor:
Type [Float32] Shape :
0 [ 1]
1 [ 28]
2 [ 28]

Details of output tensor:
Type [Int32] Shape :
0 [ 1]

Model estimate [7] training label [7]
Model estimate [2] training label [2]
:
:
Model estimate [5] training label [5]
Model estimate [4] training label [4]
Test set accuracy was 96 percent

MNIST classifier example completed successfully.

```

This example code loads the MNIST classifier, logs the sizes and datatimes of
input and output tensors, then peforms 25 inferences on a set of baked in
test data. In the future this will be extended in the future with the addition camera input
pre-processing, so that the real world written numbers can be classified in
real time.

## Training and Exporting the Model

This demo includes the Tensorflow python project that was used to train the
model and export it in the format required by Tensorflow lite micro. This can
be found in
`examples/mnist_demo/model/mnist_train_and_export_to_tfl_micro.py`. This
script defines a simple 3 layer fully connected network, then trains it using
the MNIST dataset. 

When this training is complete a TF lite flatbuffer is created containing the
network model, this is then saved as a `.cc` and `.h` file pair using the
function `write_tf_lite_micro_model()`. This is neccesary since the platforms
TFL micro targets do not have file systems, so the model needs to be compiled
directly into the binary.

Using these functions it is possible to setup a Makefile which will
automatically convert a protoBuffer model into the source files required by
TFL micro, fully automating the deployment process. 