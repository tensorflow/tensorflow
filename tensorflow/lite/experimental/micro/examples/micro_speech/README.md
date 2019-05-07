# Micro Speech Example

This examples shows how you can use TensorFlow Lite to run a 20 kilobyte neural
network model to recognize keywords in speech. It's designed to run on systems
with very small amounts of memory such as microcontrollers and DSPs. The code
itself also has a small footprint (for example around 22 kilobytes on a Cortex
M3) and only uses about 10 kilobytes of RAM for working memory, so it's able to
run on systems like an STM32F103 with only 20 kilobytes of total SRAM and 64
kilobytes of Flash.

## Table of Contents

-   [Getting Started](#getting-started)
-   [Getting Started on a Microcontroller](#getting-started-on-a-microcontroller)
-   [Calculating the Input to the Neural Network](#calculating-the-input-to-the-neural-network)
-   [Creating Your Own Model](#creating-your-own-model)

## Getting Started

To compile and test this example on a desktop Linux or MacOS machine, download
[the TensorFlow source code](https://github.com/tensorflow/tensorflow), `cd`
into the source directory from a terminal, and then run the following command:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile
```

This will take a few minutes, and downloads frameworks the code uses like
[CMSIS](https://developer.arm.com/embedded/cmsis) and
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, run:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile test_micro_speech
```

You should see a series of files get compiled, followed by some logging output
from a test, which should conclude with `~~~ALL TESTS PASSED~~~`. If you see
this, it means that a small program has been built and run that loads a trained
TensorFlow model, runs some example inputs through it, and got the expected
outputs. This particular test runs spectrograms generated from recordings of
people saying "Yes" and "No", and checks that the network correctly identifies
them.

To understand how TensorFlow Lite does this, you can look at the `TestInvoke()`
function in
[micro_speech_test.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc).
It's a fairly small amount of code, creating an interpreter, getting a handle to
a model that's been compiled into the program, and then invoking the interpreter
with the model and sample inputs.

## Getting Started on a Microcontroller

Once you have downloaded the dependencies and got the x86/Linux build working,
you can try building a version for the STM32F103 'bluepill' device. The
following command will build the test and then run it on an emulator, assuming
you have Docker installed:

*On Mac OS you need to have ARM compiler installed, one way of doing so is with
brew: brew install caskroom/cask/gcc-arm-embedded*

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=bluepill test_micro_speech
```

If you have a real device
[(see here for how to set one up)](https://github.com/google/stm32_bare_lib/tree/master/README.md)
you can then convert the ELF file into a a `.bin` format executable to load onto
it by running:

```
arm-none-eabi-objcopy \
tensorflow/lite/experimental/micro/tools/make/gen/bluepill_cortex-m3/bin/micro_speech_test \
tensorflow/lite/experimental/micro/tools/make/gen/bluepill_cortex-m3/bin/micro_speech_test.bin \
--output binary
```

## Calculating the Input to the Neural Network

The TensorFlow Lite model doesn't take in raw audio sample data. Instead it
works with spectrograms, which are two dimensional arrays that are made up of
slices of frequency information, each taken from a different time window. This
test uses spectrograms that have been pre-calculated from one-second WAV files
in the test data set. In a complete application these spectrograms would be
calculated at runtime from microphone inputs, but the code for doing that is not
yet included in this sample code.

The recipe for creating the spectrogram data is that each frequency slice is
created by running an FFT across a 30ms section of the audio sample data. The
input samples are treated as being between -1 and +1 as real values (encoded as
-32,768 and 32,767 in 16-bit signed integer samples). This results in an FFT
with 256 entries. Every sequence of six entries is averaged together, giving a
total of 43 frequency buckets in the final slice. The results are stored as
unsigned eight-bit values, where 0 represents a real number of zero, and 255
represents 127.5 as a real number. Each adjacent frequency entry is stored in
ascending memory order (frequency bucket 0 at data[0], bucket 1 at data [1],
etc). The window for the frequency analysis is then moved forward by 20ms, and
the process repeated, storing the results in the next memory row (for example
bucket 0 in this moved window would be in data[43 + 0], etc). This process
happens 49 times in total, producing a single channel image that is 43 pixels
wide, and 49 rows high. Here's an illustration of the process:

![spectrogram diagram](https://storage.googleapis.com/download.tensorflow.org/example_images/spectrogram_diagram.png)

The test data files have been generated by running the following commands:

```
bazel run tensorflow/examples/speech_commands:wav_to_features -- \
--input_wav=${HOME}/speech_commands_test_set_v0.02/yes/f2e59fea_nohash_1.wav \
--output_c_file=yes_features_data.cc \
--window_stride=20 --preprocess=average --quantize=1

bazel run tensorflow/examples/speech_commands:wav_to_features -- \
--input_wav=${HOME}/speech_commands_test_set_v0.02/no/f9643d42_nohash_4.wav \
--output_c_file=no_features_data.cc \
--window_stride=20 --preprocess=average --quantize=1
```

## Creating Your Own Model

The neural network model used in this example was built using the
[TensorFlow speech commands tutorial](https://www.tensorflow.org/tutorials/sequences/audio_recognition).

If you would like to create your own, you can start by training a model with the
following commands. Note that this will begin a full build of TensorFlow from
source; it is not currently possible to use the TensorFlow pip package. Due to
the complexity of setting up a build environment, it's easiest to run these
commands in a
[TensorFlow Docker container](https://www.tensorflow.org/install/docker). A full
build may take a couple of hours.

To begin training, run the following:

```
bazel run -c opt --copt=-mavx2 --copt=-mfma \
tensorflow/examples/speech_commands:train -- \
--model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
--wanted_words="yes,no" --silence_percentage=25 --unknown_percentage=25 --quantize=1
```

If you see a compiling error on older machines, try leaving out the `--copt`
arguments, they are just there to accelerate training on chips that support the
extensions. The training process is likely to take a couple of hours. Once it
has completed, the next step is to freeze the variables:

```
bazel run tensorflow/examples/speech_commands:freeze -- \
--model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
--wanted_words="yes,no" --quantize=1 --output_file=/tmp/tiny_conv.pb \
--start_checkpoint=/tmp/speech_commands_train/tiny_conv.ckpt-18000
```

The next step is to create a TensorFlow Lite file from the frozen graph:

```
bazel run tensorflow/lite/toco:toco -- \
--input_file=/tmp/tiny_conv.pb --output_file=/tmp/tiny_conv.tflite \
--input_shapes=1,49,40,1 --input_arrays=Reshape_1 --output_arrays='labels_softmax' \
--inference_type=QUANTIZED_UINT8 --mean_values=0 --std_values=9.8077
```

Finally, convert the file into a C source file that can be compiled into an
embedded system:

```
xxd -i /tmp/tiny_conv.tflite > /tmp/tiny_conv_micro_features_model_data.cc
```

Next, we need to update `tiny_conv_micro_features_model_data.cc` so that it is
compatible with the `micro_features` sample code.

First, open the file. The top two lines should look approximately as follows
(the exact hex values may be different):

```cpp
unsigned char _tmp_tiny_conv_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
```

You need to add the include from the following snippet, and tweak the variable
declaration. Don’t change the hex values, though:

```cpp
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h"

const unsigned char g_tiny_conv_micro_features_model_data[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
```

Next, go to the very bottom of the file and find the variable named
`_tmp_tiny_conv_tflite_len`.

```cpp
unsigned int _tmp_tiny_conv_tflite_len = 19800;
```

Change the declaration as follows, but do not change the number assigned to it,
even if your number is different from the one in this guide.

```cpp
const int g_tiny_conv_micro_features_model_data_len = 19800;
```

Finally, save the file, then copy the `tiny_conv_micro_features_model_data.cc`
file into the `micro_features/` subdirectory of your `tf_microspeech/` project.

### Creating Your Own Model With Google Cloud

If want to train your model in Google Cloud you can do so by using
pre-configured Deep Learning images.

First create the VM:

```
export IMAGE_FAMILY="tf-latest-cpu"
export ZONE="us-west1-b" # Or any other required region
export INSTANCE_NAME="model-trainer"
export INSTANCE_TYPE="n1-standard-8" # or any other instance type
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB \
        --min-cpu-platform=Intel\ Skylake
```

As soon as instance has been created you can SSH to it(as a jupyter user!):

```
gcloud compute ssh "jupyter@${INSTANCE_NAME}"
```

now install Bazel:

```
wget https://github.com/bazelbuild/bazel/releases/download/0.15.0/bazel-0.15.0-installer-linux-x86_64.sh
sudo bash ./bazel-0.15.0-installer-linux-x86_64.sh
source /usr/local/lib/bazel/bin/bazel-complete.bash
sudo ln /usr/local/bin/bazel /usr/bin/bazel
```

and finally run the build:

```
# TensorFlow already pre-baked on the image
cd src/tensorflow
bazel run -c opt --copt=-mavx2 --copt=-mfma \
tensorflow/examples/speech_commands:train -- \
--model_architecture=tiny_conv --window_stride=20 --preprocess=average \
--wanted_words="yes,no" --silence_percentage=25 --unknown_percentage=25 --quantize=1
```

After build is over follow the rest of the instrucitons from this tutorial. And
finally do not forget to remove the instance when training is done:

```
gcloud compute instances delete "${INSTANCE_NAME}" --zone="${ZONE}"
```
