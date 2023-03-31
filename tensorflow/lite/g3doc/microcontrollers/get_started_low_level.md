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
[`evaluate_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/evaluate_test.cc),
unit test which demonstrates how to run inference using TensorFlow Lite for
Microcontrollers. It loads the model and runs inference several times.

### 1. Include the library headers

To use the TensorFlow Lite for Microcontrollers library, we must include the
following header files:

```C++
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)
    provides the operations used by the interpreter to run the model.
-   [`micro_log.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_log.h)
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
provided as a C++ array. The model is defined in `hello_world_float_model_data.h` and `hello_world_float_model_data.cc` files.
The header is included with the following line:

```C++
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"
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

TF_LITE_MICRO_TEST(LoadFloatModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

We now discuss the code included in the macro above.

### 4. Set up logging

To set up logging, `micro_log.h` is used.

`MicroPrintf()` function can be used independent of the MicroErrorReporter to get
printf-like functionalitys and are common to all target platforms.

### 5. Define the input and the expected output

In the following lines, the input and the expected output are defined:

```C++
  float x = 0.0f;
  float y_true = sin(x);
```

### 6. Load a model

In the following code, the model is instantiated using data from a `char` array,
`g_hello_world_float_model_data`, which is declared in `g_hello_world_float_model_data.h`.
We then check the model to ensure its
schema version is compatible with the version we are using:

```C++
const tflite::Model* model = ::tflite::GetModel(g_hello_world_float_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  MIcroPrintf(
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 7. Instantiate operations resolver

An
[`AllOpsResolver`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/all_ops_resolver.h)
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

### 8. Allocate memory

We need to preallocate a certain amount of memory for input, output, and
intermediate arrays. This is provided as a `uint8_t` array of size
`tensor_arena_size`:

```C++
const int tensor_arena_size = 2056;
uint8_t tensor_arena[tensor_arena_size];
```

The size required will depend on the model you are using, and may need to be
determined by experimentation.

### 9. Instantiate interpreter

We create a `tflite::MicroInterpreter` instance, passing in the variables
created earlier:

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size);
```

### 10. Allocate tensors

We tell the interpreter to allocate memory from the `tensor_arena` for the
model's tensors and throw error if failed:

```C++
if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensor failed.");
    return kTfLiteError;
  }
```

### 11. Validate input shape

The `MicroInterpreter` instance can provide us with a pointer to the model's
input tensor by calling `.input(0)`, where `0` represents the first (and only)
input tensor:

```C++
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

We then inspect this tensor to confirm that it has properties what we
expect:

```C++
if (input == nullptr) {
    MicroPrintf("Input tensor in null.");
    return kTfLiteError;
  }
```

The enum value `kTfLiteFloat32` is a reference to one of the TensorFlow Lite
data types, and is defined in
[`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h).

### 12. Provide an input value

To provide an input to the model,  we set the contents of the input tensor,
as follows:

```C++
input->data.f[0] = x;
```

In this case, we input a quantized input `x`.

### 13. Run the model

To run the model, we can call `Invoke()` on our `tflite::MicroInterpreter`
instance:

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
    MicroPrintf("Interpreter invocation failed.");
    return kTfLiteError;
  }
```

We can check the return value, a `TfLiteStatus`, to determine if the run was
successful. The possible values of `TfLiteStatus`, defined in
[`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h),
are `kTfLiteOk` and `kTfLiteError`.


### 14. Obtain the output

The model's output tensor can be obtained by calling `output(0)` on the
`tflite::MicroInterpreter`, where `0` represents the first (and only) output
tensor.

In the example, the model's output is a single floating point value contained
within a 2D tensor:

```C++
TfLiteTensor* output = interpreter.output(0);
```

We can read the value directly from the output tensor and assert that it is what
we expect:

```C++
// Obtain the output value from the tensor
float y_pred = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
float epsilon = 0.05f;
  if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
  }
```

### 15. Run inference again

The remainder of the code runs inference several more times. In each instance,
we assign a value to the input tensor, invoke the interpreter, and read the
result from the output tensor:

```C++
    x = 1.f;
    y_true = sin(x);
    input->data.f[0] = x;
    interpreter.Invoke();
    y_pred = output->data.f[0];
    if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
    }

    x = 3.f;
    y_true = sin(x);
    input->data.f[0] = x;
    interpreter.Invoke();
    y_pred = output->data.f[0];
    if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
    }

    x = 5.f;
    y_true = sin(x);
    input->data.f[0] = x;
    interpreter.Invoke();
    y_pred = output->data.f[0];
    if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
    }
```


