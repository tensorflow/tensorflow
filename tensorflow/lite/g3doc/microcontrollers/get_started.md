# Get started with microcontrollers

This document will help you start working with TensorFlow Lite for
Microcontrollers.

## Sample code

To get started, you can explore the following example:

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech">Micro
speech example</a>

This example uses a simple
[audio recognition model](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
to identify keywords in speech. The sample code captures audio from a device's
microphones. The model classifies this audio in real time, determining whether
the word "yes" or "no" has been spoken.

The sample works end-to-end (including audio capture and inference) on the
following platforms:

-   [SparkFun Edge Powered by TensorFlow (Apollo3 Blue)](https://www.sparkfun.com/products/15170)
-   [STM32F746G Discovery Board](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
-   Mac OS X

### SparkFun Edge

If you need a device to get started, we recommend the
[SparkFun Edge Powered by TensorFlow](https://www.sparkfun.com/products/15170).
It was designed in conjunction with the TensorFlow Lite team to offer a flexible
platform for experimenting with deep learning on microcontrollers.

To get started using the Edge board, we recommend following
[Machine learning on a microcontroller with SparkFun TensorFlow](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow),
a codelab that introduces you to the development workflow.

## Workflow

Using TensorFlow Lite for Microcontrollers involves four major steps:

1.  Create or find a model architecture.
2.  Train a model.
3.  Convert the model.
4.  Write code to run inference.

The first three steps are covered in the guide
[Build and convert models](build_convert.md). The sample code comes with a
pretrained model, and includes scripts to train a model that recognizes
different spoken words. Instructions on training are in
[README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/README.md#creating-your-own-model).

In this document, we will focus on the code that will feed processed audio data
into the model and execute it, resulting in a prediction of which word was
spoken. This process is called *inference*.

## Run inference

The sample's
[main.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/main.cc)
contains the code that runs inference. We'll now walk through the key parts.

### Includes

To use the library, we must include the following header files:

```C++
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h)
    provides the operations used by the interpreter to run the model.
-   [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_error_reporter.h)
    outputs debug information.
-   [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_interpreter.h)
    contains code to handle and run models.
-   [`schema_generated.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_generated.h)
    contains the schema for the TensorFlow Lite
    [`FlatBuffer`](https://google.github.io/flatbuffers/) model file format.
-   [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h)
    provides versioning information for the TensorFlow Lite schema.

The sample also includes some other files. These are the most significant:

```C++
#include "tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h"
```

-   [`feature_provider.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/feature_provider.h)
    contains code to extract features from the audio stream to input to the
    model.
-   [`tiny_conv_micro_features_model_data.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h)
    contains the model stored as a `char` array. Read
    [Build and convert models](build_convert.md) to learn how to convert a
    TensorFlow Lite model into this format.
-   [`micro_model_settings.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h)
    defines various constants related to the model.

### Set up logging

To set up logging, a `tflite::ErrorReporter` pointer is created using a pointer
to a `tflite::MicroErrorReporter` instance:

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

This variable will be passed into the interpreter, which allows it to write
logs. Since microcontrollers often have a variety of mechanisms for logging, the
implementation of `tflite::MicroErrorReporter` is designed to be customized for
your particular device.

### Load a model

In the following code, the model is instantiated from a `char` array,
`g_tiny_conv_micro_features_model_data` (to learn how this is created, see
[Build and convert models](build_convert.md)). We then check the model to ensure
its schema version is compatible with the version we are using:

```C++
const tflite::Model* model =
    ::tflite::GetModel(g_tiny_conv_micro_features_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
  return 1;
}
```

### Instantiate operations resolver

An
[`AllOpsResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h)
instance is required by the interpreter to access TensorFlow operations. This
class can be extended to add custom operations to your project:

```C++
tflite::ops::micro::AllOpsResolver resolver;
```

### Allocate memory

We need to preallocate a certain amount of memory for input, output, and
intermediate arrays. This is provided as a `uint8_t` array of size
`tensor_arena_size`, which is passed into a `tflite::SimpleTensorAllocator`
instance:

```C++
const int tensor_arena_size = 10 * 1024;
uint8_t tensor_arena[tensor_arena_size];
tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                               tensor_arena_size);
```

Note: The size required will depend on the model you are using, and may need to
be determined by experimentation.

### Instantiate interpreter

We create a `tflite::MicroInterpreter` instance, passing in the variables
created earlier:

```C++
tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                     error_reporter);
```

### Validate input shape

The `MicroInterpreter` instance can provide us with a pointer to the model's
input tensor by calling `.input(0)`, where `0` represents the first (and only)
input tensor. We inspect this tensor to confirm that its shape and type are what
we are expecting:

```C++
TfLiteTensor* model_input = interpreter.input(0);
if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
    (model_input->dims->data[1] != kFeatureSliceCount) ||
    (model_input->dims->data[2] != kFeatureSliceSize) ||
    (model_input->type != kTfLiteUInt8)) {
  error_reporter->Report("Bad input tensor parameters in model");
  return 1;
}
```

In this snippet, the variables `kFeatureSliceCount` and `kFeatureSliceSize`
relate to properties of the input and are defined in
[`micro_model_settings.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h).
The enum value `kTfLiteUInt8` is a reference to one of the TensorFlow Lite data
types, and is defined in
[`c_api_internal.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_internal.h).

### Generate features

The data we input to our model must be generated from the microcontroller's
audio input. The `FeatureProvider` class defined in
[`feature_provider.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/feature_provider.h)
captures audio and converts it into a set of features that will be passed into
the model. When it is instantiated, we use the `TfLiteTensor` obtained earlier
to pass in a pointer to the input array. This is used by the `FeatureProvider`
to populate the input data that will be passed into the model:

```C++
  FeatureProvider feature_provider(kFeatureElementCount,
                                   model_input->data.uint8);
```

The following code causes the `FeatureProvider` to generate a set of features
from the most recent second of audio and populate the input tensor:

```C++
TfLiteStatus feature_status = feature_provider.PopulateFeatureData(
    error_reporter, previous_time, current_time, &how_many_new_slices);
```

In the sample, feature generation and inference happens in a loop, so the device
is constantly capturing and processing new audio.

If you are writing your own program, you will likely generate features in a
different way, but you will always populate the input tensor with data before
running the model.

### Run the model

To run the model, we can call `Invoke()` on our `tflite::MicroInterpreter`
instance:

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  error_reporter->Report("Invoke failed");
  return 1;
}
```

We can check the return value, a `TfLiteStatus`, to determine if the run was
successful. The possible values of `TfLiteStatus`, defined in
[`c_api_internal.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_internal.h),
are `kTfLiteOk` and `kTfLiteError`.

### Obtain the output

The model's output tensor can be obtained by calling `output(0)` on the
`tflite::MicroIntepreter`, where `0` represents the first (and only) output
tensor.

In the sample, the output is an array representing the probability of the input
belonging to various classes (representing "yes", "no", "unknown", and
"silence"). Since they are in a set order, we can use simple logic to determine
which class has the highest probability:

```C++
    TfLiteTensor* output = interpreter.output(0);
    uint8_t top_category_score = 0;
    int top_category_index;
    for (int category_index = 0; category_index < kCategoryCount;
         ++category_index) {
      const uint8_t category_score = output->data.uint8[category_index];
      if (category_score > top_category_score) {
        top_category_score = category_score;
        top_category_index = category_index;
      }
    }
```

Elsewhere in the sample, a more sophisticated algorithm is used to smooth
recognition results across a number of frames. This is defined in
[recognize_commands.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h).
The same technique can be used to improve reliability when processing any
continuous stream of data.

## Build the sample

The sample contains build scripts that will download all required dependencies
and compile a binary that can be run on a device.

Note: The build process has been tested on MacOS and Linux, but not on Windows.

To build the sample, take the following steps:

1.  Clone the TensorFlow repository from GitHub to a convenient place.

    ```bash
    git clone --depth 1 https://github.com/tensorflow/tensorflow.git
    ```

1.  Enter the directory that was created in the previous step.

    ```bash
    cd tensorflow
    ```

1.  If you are using MacOS, run the following command. If you are using Linux,
    you do not need to do this.

    ```bash
    PATH=tensorflow/lite/experimental/micro/tools/make/downloads/gcc_embedded/bin/:$PATH
    ```

1.  To download all of the required dependencies and initiate the build process,
    issue the following command. You can set `TARGET` depending on which
    platform you want to build for. Explore
    [`targets/`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/tools/make/targets)
    for the current options.

    ```bash
    make -f tensorflow/lite/experimental/micro/tools/make/Makefile
    TARGET=sparkfun_edge micro_speech_bin
    ```

## Next steps

Once you have built and run the sample, read the following documents:

*   Learn how to work with models in
    [Build and convert models](build_convert.md).
*   Learn more about the C++ library in
    [Understand the C++ library](library.md).
