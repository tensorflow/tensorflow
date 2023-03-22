# Get started with microcontrollers

This document explains how to train a model and run inference using a
microcontroller.

## The Hello World example

The
[Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world)
example is designed to demonstrate the absolute basics of using TensorFlow Lite
for Microcontrollers. We train and run a model that replicates a sine function,
i.e, it takes a single number as its input, and outputs the number's
[sine](https://en.wikipedia.org/wiki/Sine) value. When deployed to the
microcontroller, its predictions are used to either blink LEDs or control an
animation.

The end-to-end workflow involves the following steps:

1.  [Train a model](#train_a_model) (in Python): A jupyter notebook to train,
    convert and optimize a model for on-device use.
2.  [Run inference](#run_inference) (in C++ 17): An end-to-end unit test that
    runs inference on the model using the [C++ library](library.md).

## Get a supported device

The example application we'll be using has been tested on the following devices:

*   [Arduino Nano 33 BLE Sense](https://store-usa.arduino.cc/products/arduino-nano-33-ble-sense-with-headers)
    (using Arduino IDE)
*   [SparkFun Edge](https://www.sparkfun.com/products/15170) (building directly
    from source)
*   [STM32F746 Discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
    (using Mbed)
*   [Adafruit EdgeBadge](https://www.adafruit.com/product/4400) (using Arduino
    IDE)
*   [Adafruit TensorFlow Lite for Microcontrollers Kit](https://www.adafruit.com/product/4317)
    (using Arduino IDE)
*   [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all)
    (using Arduino IDE)
*   [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview)
    (using ESP IDF)
*   [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview)
    (using ESP IDF)

Learn more about supported platforms in
[TensorFlow Lite for Microcontrollers](index.md).

## Train a model

Note: You can skip this section and use the trained model included in the
example code.

Use Google Colaboratory to
[train your own model](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb).
For more details, refer to the `README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/train/README.md">Hello
World Training README.md</a>

## Run inference

To run the model on your device, we will walk through the instructions in the
`README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello
World README.md</a>

The following sections walk through the example's
[`hello_world_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc),
unit test which demonstrates how to run inference using TensorFlow Lite for
Microcontrollers. It loads the model and runs inference several times.

### 1. Include the library headers

To use the TensorFlow Lite for Microcontrollers library, we must include the
following header files:

```C++
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)
    provides the operations used by the interpreter to run the model.
-   [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_error_reporter.h)
    outputs debug information.
-   [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h)
    contains code to load and run models.
-   [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h)
    contains the schema for the TensorFlow Lite
    [`FlatBuffer`](https://google.github.io/flatbuffers/) model file format.
-   [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h)
    provides versioning information for the TensorFlow Lite schema.

### 2. Include the model header

The TensorFlow Lite for Microcontrollers interpreter expects the model to be
provided as a C++ array. The model is defined in `model.h` and `model.cc` files.
The header is included with the following line:

```C++
#include "tensorflow/lite/micro/examples/hello_world/model.h"
```

### 3. Include the unit test framework header

In order to create a unit test, we include the TensorFlow Lite for
Microcontrollers unit test framework by including the following line:

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

The test is defined using the following macros:

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

We now discuss the code included in the macro above.

### 4. Set up logging

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

### 5. Load a model

In the following code, the model is instantiated using data from a `char` array,
`g_model`, which is declared in `model.h`. We then check the model to ensure its
schema version is compatible with the version we are using:

```C++
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 6. Instantiate operations resolver

An
[`AllOpsResolver`](github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)
instance is declared. This will be used by the interpreter to access the
operations that are used by the model:

```C++
tflite::AllOpsResolver resolver;
```

The `AllOpsResolver` loads all of the operations available in TensorFlow Lite
for Microcontrollers, which uses a lot of memory. Since a given model will only
use a subset of these operations, it's recommended that real world applications
load only the operations that are needed.

This is done using a different class, `MicroMutableOpResolver`. You can see how
to use it in the *Micro speech* example's
[`micro_speech_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc).

### 7. Allocate memory

We need to preallocate a certain amount of memory for input, output, and
intermediate arrays. This is provided as a `uint8_t` array of size
`tensor_arena_size`:

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

The size required will depend on the model you are using, and may need to be
determined by experimentation.

### 8. Instantiate interpreter

We create a `tflite::MicroInterpreter` instance, passing in the variables
created earlier:

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. Allocate tensors

We tell the interpreter to allocate memory from the `tensor_arena` for the
model's tensors:

```C++
interpreter.AllocateTensors();
```

### 10. Validate input shape

The `MicroInterpreter` instance can provide us with a pointer to the model's
input tensor by calling `.input(0)`, where `0` represents the first (and only)
input tensor:

```C++
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

We then inspect this tensor to confirm that its shape and type are what we are
expecting:

```C++
// Make sure the input has the properties we expect
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// The input is a 32 bit floating point value
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
```

The enum value `kTfLiteFloat32` is a reference to one of the TensorFlow Lite
data types, and is defined in
[`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h).

### 11. Provide an input value

To provide an input to the model, we set the contents of the input tensor, as
follows:

```C++
input->data.f[0] = 0.;
```

In this case, we input a floating point value representing `0`.

### 12. Run the model

To run the model, we can call `Invoke()` on our `tflite::MicroInterpreter`
instance:

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
}
```

We can check the return value, a `TfLiteStatus`, to determine if the run was
successful. The possible values of `TfLiteStatus`, defined in
[`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h),
are `kTfLiteOk` and `kTfLiteError`.

The following code asserts that the value is `kTfLiteOk`, meaning inference was
successfully run.

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. Obtain the output

The model's output tensor can be obtained by calling `output(0)` on the
`tflite::MicroInterpreter`, where `0` represents the first (and only) output
tensor.

In the example, the model's output is a single floating point value contained
within a 2D tensor:

```C++
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);
```

We can read the value directly from the output tensor and assert that it is what
we expect:

```C++
// Obtain the output value from the tensor
float value = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 14. Run inference again

The remainder of the code runs inference several more times. In each instance,
we assign a value to the input tensor, invoke the interpreter, and read the
result from the output tensor:

```C++
input->data.f[0] = 1.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);

input->data.f[0] = 3.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);

input->data.f[0] = 5.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);
```

### 15. Read the application code

Once you have walked through this unit test, you should be able to understand
the example's application code, located in
[`main_functions.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/main_functions.cc).
It follows a similar process, but generates an input value based on how many
inferences have been run, and calls a device-specific function that displays the
model's output to the user.
