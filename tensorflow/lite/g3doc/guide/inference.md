# TensorFlow Lite inference

The term *inference* refers to the process of executing a TensorFlow Lite model
on-device in order to make predictions based on input data. To perform an
inference with a TensorFlow Lite model, you must run it through an
*interpreter*. The TensorFlow Lite interpreter is designed to be lean and fast.
The interpreter uses a static graph ordering and a custom (less-dynamic) memory
allocator to ensure minimal load, initialization, and execution latency.

This page describes how to access to the TensorFlow Lite interpreter and perform
an inference using C++, Java, and Python, plus links to other resources for each
[supported platform](#supported-platforms).

[TOC]

## Important concepts

TensorFlow Lite inference typically follows the following steps:

1.  **Loading a model**

    You must load the `.tflite` model into memory, which contains the model's
    execution graph.

1.  **Transforming data**

    Raw input data for the model generally does not match the input data format
    expected by the model. For example, you might need to resize an image or
    change the image format to be compatible with the model.

1.  **Running inference**

    This step involves using the TensorFlow Lite API to execute the model. It
    involves a few steps such as building the interpreter, and allocating
    tensors, as described in the following sections.

1.  **Interpreting output**

    When you receive results from the model inference, you must interpret the
    tensors in a meaningful way that's useful in your application.

    For example, a model might return only a list of probabilities. It's up to
    you to map the probabilities to relevant categories and present it to your
    end-user.

## Supported platforms

TensorFlow inference APIs are provided for most common mobile/embedded platforms
such as [Android](#android-platform), [iOS](#ios-platform) and
[Linux](#linux-platform), in multiple programming languages.

In most cases, the API design reflects a preference for performance over ease of
use. TensorFlow Lite is designed for fast inference on small devices, so it
should be no surprise that the APIs try to avoid unnecessary copies at the
expense of convenience. Similarly, consistency with TensorFlow APIs was not an
explicit goal and some variance between languages is to be expected.

Across all libraries, the TensorFlow Lite API enables you to load models, feed
inputs, and retrieve inference outputs.

### Android Platform

On Android, TensorFlow Lite inference can be performed using either Java or C++
APIs. The Java APIs provide convenience and can be used directly within your
Android Activity classes. The C++ APIs offer more flexibility and speed, but may
require writing JNI wrappers to move data between Java and C++ layers.

See below for details about using [C++](#load-and-run-a-model-in-c) and
[Java](#load-and-run-a-model-in-java), or follow the
[Android quickstart](../android) for a tutorial and example code.

#### TensorFlow Lite Android wrapper code generator

Note: TensorFlow Lite wrapper code generator is in experimental (beta) phase and
it currently only supports Android.

For TensorFlow Lite model enhanced with [metadata](../inference_with_metadata/overview),
developers can use the TensorFlow Lite Android wrapper code generator to create
platform specific wrapper code. The wrapper code removes the need to interact
directly with `ByteBuffer` on Android. Instead, developers can interact with the
TensorFlow Lite model with typed objects such as `Bitmap` and `Rect`. For more
information, please refer to the
[TensorFlow Lite Android wrapper code generator](../inference_with_metadata/codegen.md).

### iOS Platform

On iOS, TensorFlow Lite is available with native iOS libraries written in
[Swift](https://www.tensorflow.org/code/tensorflow/lite/swift)
and
[Objective-C](https://www.tensorflow.org/code/tensorflow/lite/objc).
You can also use
[C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h)
directly in Objective-C codes.

See below for details about using [Swift](#load-and-run-a-model-in-swift),
[Objective-C](#load-and-run-a-model-in-objective-c) and the
[C API](#using-c-api-in-objective-c-code), or follow the
[iOS quickstart](ios.md) for a tutorial and example code.

### Linux Platform

On Linux platforms (including [Raspberry Pi](build_arm)), you can run
inferences using TensorFlow Lite APIs available in
[C++](#load-and-run-a-model-in-c) and [Python](#load-and-run-a-model-in-python),
as shown in the following sections.

## Running a model

Running a TensorFlow Lite model involves a few simple steps:

1.  Load the model into memory.
2.  Build an `Interpreter` based on an existing model.
3.  Set input tensor values. (Optionally resize input tensors if the predefined
    sizes are not desired.)
4.  Invoke inference.
5.  Read output tensor values.

Following sections describe how these steps can be done in each language.

## Load and run a model in Java

*Platform: Android*

The Java API for running an inference with TensorFlow Lite is primarily designed
for use with Android, so it's available as an Android library dependency:
`org.tensorflow:tensorflow-lite`.

In Java, you'll use the `Interpreter` class to load a model and drive model
inference. In many cases, this may be the only API you need.

You can initialize an `Interpreter` using a `.tflite` file:

```java
public Interpreter(@NotNull File modelFile);
```

Or with a `MappedByteBuffer`:

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

In both cases, you must provide a valid TensorFlow Lite model or the API throws
`IllegalArgumentException`. If you use `MappedByteBuffer` to initialize an
`Interpreter`, it must remain unchanged for the whole lifetime of the
`Interpreter`.

The preferred way to run inference on a model is to use signatures -
Available for models converted starting Tensorflow 2.5

```Java
try (Interpreter interpreter = new Interpreter(file_of_tensorflowlite_model)) {
  Map<String, Object> inputs = new HashMap<>();
  inputs.put("input_1", input1);
  inputs.put("input_2", input2);
  Map<String, Object> outputs = new HashMap<>();
  outputs.put("output_1", output1);
  interpreter.runSignature(inputs, outputs, "mySignature");
}
```

The `runSignature` method takes three arguments:

-   **Inputs** : map for inputs from input name in the signature to an input
    object.

-   **Outputs** : map for output mapping from output name in signature to output
    data.

-   **Signature Name** [optional]: Signature name (Can be left empty if the
    model has single signature).

Another way to run an inference when the model doesn't
have a defined signatures.
Simply call `Interpreter.run()`. For example:

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

The `run()` method takes only one input and returns only one output. So if your
model has multiple inputs or multiple outputs, instead use:

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

In this case, each entry in `inputs` corresponds to an input tensor and
`map_of_indices_to_outputs` maps indices of output tensors to the corresponding
output data.

In both cases, the tensor indices should correspond to the values you gave to
the [TensorFlow Lite Converter](../models/convert/) when you created the model. Be
aware that the order of tensors in `input` must match the order given to the
TensorFlow Lite Converter.

The `Interpreter` class also provides convenient functions for you to get the
index of any model input or output using an operation name:

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

If `opName` is not a valid operation in the model, it throws an
`IllegalArgumentException`.

Also beware that `Interpreter` owns resources. To avoid memory leak, the
resources must be released after use by:

```java
interpreter.close();
```

For an example project with Java, see the
[Android image classification sample](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

### Supported data types (in Java)

To use TensorFlow Lite, the data types of the input and output tensors must be
one of the following primitive types:

*   `float`
*   `int`
*   `long`
*   `byte`

`String` types are also supported, but they are encoded differently than the
primitive types. In particular, the shape of a string Tensor dictates the number
and arrangement of strings in the Tensor, with each element itself being a
variable length string. In this sense, the (byte) size of the Tensor cannot be
computed from the shape and type alone, and consequently strings cannot be
provided as a single, flat `ByteBuffer` argument. You can see some examples in
this [page](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter).

If other data types, including boxed types like `Integer` and `Float`, are used,
an `IllegalArgumentException` will be thrown.

#### Inputs

Each input should be an array or multi-dimensional array of the supported
primitive types, or a raw `ByteBuffer` of the appropriate size. If the input is
an array or multi-dimensional array, the associated input tensor will be
implicitly resized to the array's dimensions at inference time. If the input is
a ByteBuffer, the caller should first manually resize the associated input
tensor (via `Interpreter.resizeInput()`) before running inference.

When using `ByteBuffer`, prefer using direct byte buffers, as this allows the
`Interpreter` to avoid unnecessary copies. If the `ByteBuffer` is a direct byte
buffer, its order must be `ByteOrder.nativeOrder()`. After it is used for a
model inference, it must remain unchanged until the model inference is finished.

#### Outputs

Each output should be an array or multi-dimensional array of the supported
primitive types, or a ByteBuffer of the appropriate size. Note that some models
have dynamic outputs, where the shape of output tensors can vary depending on
the input. There's no straightforward way of handling this with the existing
Java inference API, but planned extensions will make this possible.

## Load and run a model in Swift

*Platform: iOS*

The
[Swift API](https://www.tensorflow.org/code/tensorflow/lite/swift)
is available in `TensorFlowLiteSwift` Pod from Cocoapods.

First, you need to import `TensorFlowLite` module.

```swift
import TensorFlowLite
```

```swift
// Getting model path
guard
  let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")
else {
  // Error handling...
}

do {
  // Initialize an interpreter with the model.
  let interpreter = try Interpreter(modelPath: modelPath)

  // Allocate memory for the model's input `Tensor`s.
  try interpreter.allocateTensors()

  let inputData: Data  // Should be initialized

  // input data preparation...

  // Copy the input data to the input `Tensor`.
  try self.interpreter.copy(inputData, toInputAt: 0)

  // Run inference by invoking the `Interpreter`.
  try self.interpreter.invoke()

  // Get the output `Tensor`
  let outputTensor = try self.interpreter.output(at: 0)

  // Copy output to `Data` to process the inference results.
  let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
  let outputData =
        UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)
  outputTensor.data.copyBytes(to: outputData)

  if (error != nil) { /* Error handling... */ }
} catch error {
  // Error handling...
}
```

## Load and run a model in Objective-C

*Platform: iOS*

The
[Objective-C API](https://www.tensorflow.org/code/tensorflow/lite/objc)
is available in `TensorFlowLiteObjC` Pod from Cocoapods.

First, you need to import `TensorFlowLite` module.

```objc
@import TensorFlowLite;
```

```objc
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model"
                                                      ofType:@"tflite"];
NSError *error;

// Initialize an interpreter with the model.
TFLInterpreter *interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                  error:&error];
if (error != nil) { /* Error handling... */ }

// Allocate memory for the model's input `TFLTensor`s.
[interpreter allocateTensorsWithError:&error];
if (error != nil) { /* Error handling... */ }

NSMutableData *inputData;  // Should be initialized
// input data preparation...

// Get the input `TFLTensor`
TFLTensor *inputTensor = [interpreter inputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy the input data to the input `TFLTensor`.
[inputTensor copyData:inputData error:&error];
if (error != nil) { /* Error handling... */ }

// Run inference by invoking the `TFLInterpreter`.
[interpreter invokeWithError:&error];
if (error != nil) { /* Error handling... */ }

// Get the output `TFLTensor`
TFLTensor *outputTensor = [interpreter outputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy output to `NSData` to process the inference results.
NSData *outputData = [outputTensor dataWithError:&error];
if (error != nil) { /* Error handling... */ }
```

### Using C API in Objective-C code

Currently Objective-C API does not support delegates. In order to use delegates
with Objective-C code, you need to directly call underlying
[C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h).

```c
#include "tensorflow/lite/c/c_api.h"
```

```c
TfLiteModel* model = TfLiteModelCreateFromFile([modelPath UTF8String]);
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

// Create the interpreter.
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

// Allocate tensors and populate the input tensor data.
TfLiteInterpreterAllocateTensors(interpreter);
TfLiteTensor* input_tensor =
    TfLiteInterpreterGetInputTensor(interpreter, 0);
TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                           input.size() * sizeof(float));

// Execute inference.
TfLiteInterpreterInvoke(interpreter);

// Extract the output tensor data.
const TfLiteTensor* output_tensor =
    TfLiteInterpreterGetOutputTensor(interpreter, 0);
TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                         output.size() * sizeof(float));

// Dispose of the model and interpreter objects.
TfLiteInterpreterDelete(interpreter);
TfLiteInterpreterOptionsDelete(options);
TfLiteModelDelete(model);
```

## Load and run a model in C++

*Platforms: Android, iOS, and Linux*

Note: C++ API on iOS is only available when using bazel.

In C++, the model is stored in
[`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html)
class. It encapsulates a TensorFlow Lite model and you can build it in a couple
of different ways, depending on where the model is stored:

```c++
class FlatBufferModel {
  // Build a model based on a file. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter);

  // Build a model based on a pre-loaded flatbuffer. The caller retains
  // ownership of the buffer and should keep it alive until the returned object
  // is destroyed. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* buffer,
      size_t buffer_size,
      ErrorReporter* error_reporter);
};
```

Note: If TensorFlow Lite detects the presence of the
[Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks), it
will automatically try to use shared memory to store the `FlatBufferModel`.

Now that you have the model as a `FlatBufferModel` object, you can execute it
with an
[`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html).
A single `FlatBufferModel` can be used simultaneously by more than one
`Interpreter`.

Caution: The `FlatBufferModel` object must remain valid until all instances of
`Interpreter` using it have been destroyed.

The important parts of the `Interpreter` API are shown in the code snippet
below. It should be noted that:

*   Tensors are represented by integers, in order to avoid string comparisons
    (and any fixed dependency on string libraries).
*   An interpreter must not be accessed from concurrent threads.
*   Memory allocation for input and output tensors must be triggered by calling
    `AllocateTensors()` right after resizing tensors.

The simplest usage of TensorFlow Lite with C++ looks like this:

```c++
// Load the model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);

// Build the interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
interpreter->AllocateTensors();

float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

interpreter->Invoke();

float* output = interpreter->typed_output_tensor<float>(0);
```

For more example code, see
[`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc)
and
[`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc).

## Load and run a model in Python

*Platform: Linux*

The Python API for running an inference is provided in the `tf.lite` module.
From which, you mostly need only
[`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)
to load a model and run an inference.

The following example shows how to use the Python interpreter to load a
`.tflite` file and run inference with random input data:

This example is recommended if you're converting from SavedModel with a defined
SignatureDef.
Available starting from TensorFlow 2.5

```python
class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def add(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 4}


SAVED_MODEL_PATH = 'content/saved_models/test_variable'
TFLITE_FILE_PATH = 'content/test_variable.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH,
    signatures={'my_signature':module.add.get_concrete_function()})

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'])
```

Another example if the model doesn't have SignatureDefs defined.

```python
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

As an alternative to loading the model as a pre-converted `.tflite` file, you
can combine your code with the
[TensorFlow Lite Converter Python API](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter)
(`tf.lite.TFLiteConverter`), allowing you to convert your Keras model into
the TensorFlow Lite format and then run inference:

```python
import numpy as np
import tensorflow as tf

img = tf.keras.Input(shape=(64, 64, 3), name="img")
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

# Convert to TF Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.Model(inputs=[img], outputs=[out]))
tflite_model = converter.convert()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Continue to get tensors and so forth, as shown above...
```

For more Python sample code, see
[`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

Tip: Run `help(tf.lite.Interpreter)` in the Python terminal to get detailed
documentation about the interpreter.

## Run inference with dynamic shape model

If you want to run a model with dynamic input shape,
*resize the input shape* before running inference.
Otherwise, the `None` shape in Tensorflow models will be replaced by a
placeholder of `1` in TFLite models.

The following examples show how to resize the input shape before
running inference in different languages.
All the examples assume that the input shape is defined as `[1/None, 10]`, and
need to be resized to `[3, 10]`.

<section class="tabs">

###### C++ {.new-tab}

```c++
// Resize input tensors before allocate tensors
interpreter->ResizeInputTensor(/*tensor_index=*/0, std::vector<int>{3,10});
interpreter->AllocateTensors();
```
###### Python {.new-tab}

```python
# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
  
# Resize input shape for dynamic shape model and allocate tensor
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], [3, 10])
interpreter.allocate_tensors()
  
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

</section>

## Supported operations

TensorFlow Lite supports a subset of TensorFlow operations with some
limitations. For full list of operations and limitations see
[TF Lite Ops page](https://www.tensorflow.org/mlir/tfl_ops).
