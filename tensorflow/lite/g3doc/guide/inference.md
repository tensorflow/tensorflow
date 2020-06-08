# TensorFlow Lite inference

The term *inference* refers to the process of executing a TensorFlow Lite model
on-device in order to make predictions based on input data. To perform an
inference with a TensorFlow Lite model, you must run it through an
*interpreter*. The TensorFlow Lite interpreter is designed to be lean and fast.
The interpreter uses a static graph ordering and a custom (less-dynamic) memory
allocator to ensure minimal load, initialization, and execution latency.

This page describes how to access to the TensorFlow Lite interpreter and
perform an inference using C++, Java, and Python, plus links to other resources
for each [supported platform](#supported-platforms).

[TOC]

## Important concepts

TensorFlow Lite inference typically follows the following steps:

1. **Loading a model**

   You must load the `.tflite` model into memory, which contains the model's
   execution graph.

1. **Transforming data**

   Raw input data for the model generally does not match the input data format
   expected by the model. For example, you might need to resize an image or
   change the image format to be compatible with the model.

1. **Running inference**

   This step involves using the TensorFlow Lite API to execute the model. It
   involves a few steps such as building the interpreter, and allocating
   tensors, as described in the following sections.

1. **Interpreting output**

   When you receive results from the model inference, you must interpret the
   tensors in a meaningful way that's useful in your application.

   For example, a model might return only a list of probabilities. It's up to
   you to map the probabilities to relevant categories and present it to your
   end-user.

## Supported platforms

TensorFlow inference APIs are provided for most common mobile/embedded platforms
such as Android, iOS and Linux, in multiple programming languages.

In most cases, the API design reflects a preference for performance over ease of
use. TensorFlow Lite is designed for fast inference on small devices, so it
should be no surprise that the APIs try to avoid unnecessary copies at the
expense of convenience. Similarly, consistency with TensorFlow APIs was not an
explicit goal and some variance between languages is to be expected.

Across all libraries, the TensorFlow Lite API enables you to load models,
feed inputs, and retrieve inference outputs.

### Android

On Android, TensorFlow Lite inference can be performed using either Java or C++
APIs. The Java APIs provide convenience and can be used directly within your
Android Activity classes. The C++ APIs offer more flexibility and speed, but may
require writing JNI wrappers to move data between Java and C++ layers.

See below for details about using C++ and Java, or
follow the [Android quickstart](android.md) for a tutorial and example code.

#### TensorFlow Lite Android wrapper code generator

Note: TensorFlow Lite wrapper code generator is in experimental (beta) phase and
it currently only supports Android.

For TensorFlow Lite model enhanced with [metadata](../convert/metadata.md),
developers can use the TensorFlow Lite Android wrapper code generator to create
platform specific wrapper code. The wrapper code removes the need to interact
directly with `ByteBuffer` on Android. Instead, developers can interact with the
TensorFlow Lite model with typed objects such as `Bitmap` and `Rect`. For more
information, please refer to the
[TensorFlow Lite Android wrapper code generator](codegen.md).

### iOS

On iOS, TensorFlow Lite is available with native iOS libraries written in
[Swift](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift)
and
[Objective-C](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc).

This page doesn't include a discussion for about these languages, so you should
refer to the [iOS quickstart](ios.md) for a tutorial and example code.

### Linux

On Linux platforms (including [Raspberry Pi](build_rpi.md)), you can run
inferences using TensorFlow Lite APIs available in C++ and Python, as shown
in the following sections.


## Load and run a model in C++

Running a TensorFlow Lite model with C++ involves a few simple steps:

  1. Load the model into memory as a `FlatBufferModel`.
  2. Build an `Interpreter` based on an existing `FlatBufferModel`.
  3. Set input tensor values. (Optionally resize input tensors if the
     predefined sizes are not desired.)
  4. Invoke inference.
  5. Read output tensor values.

The [`FlatBufferModel`](
https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html)
class encapsulates a TensorFlow Lite model and you can
build it in a couple of different ways, depending on where the model is stored:

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

Note: If TensorFlow Lite detects the presence of the [Android NNAPI](
https://developer.android.com/ndk/guides/neuralnetworks), it will
automatically try to use shared memory to store the `FlatBufferModel`.

Now that you have the model as a `FlatBufferModel` object, you can execute it
with an [`Interpreter`](
https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html).
A single `FlatBufferModel` can be used
simultaneously by more than one `Interpreter`.

Caution: The `FlatBufferModel` object must remain valid until
all instances of `Interpreter` using it have been destroyed.

The important parts of the `Interpreter` API are shown in the
code snippet below. It should be noted that:

  * Tensors are represented by integers, in order to avoid string comparisons
    (and any fixed dependency on string libraries).
  * An interpreter must not be accessed from concurrent threads.
  * Memory allocation for input and output tensors must be triggered
    by calling `AllocateTensors()` right after resizing tensors.

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

For more example code, see [`minimal.cc`](
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc)
and [`label_image.cc`](
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc).


## Load and run a model in Java

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
`IllegalArgumentException`. If you use `MappedByteBuffer` to
initialize an `Interpreter`, it must remain unchanged for the whole lifetime
of the `Interpreter`.

To then run an inference with the model, simply call `Interpreter.run()`.
For example:

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
the [TensorFlow Lite Converter](../convert/) when you created the model.
Be aware that the order of tensors in `input` must match the
order given to the TensorFlow Lite Converter.

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

For an example project with Java, see the [Android image classification sample](
https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

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
provided as a single, flat `ByteBuffer` argument.

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


## Load and run a model in Python

The Python API for running an inference is provided in the `tf.lite`
module. From which, you mostly need only [`tf.lite.Interpreter`](
https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) to load
a model and run an inference.

The following example shows how to use the Python interpreter to load a
`.tflite` file and run inference with random input data:

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

Alternatively to loading the model as a pre-converted `.tflite` file, you can
combine your code with the
[TensorFlow Lite Converter Python API](../convert/python_api.md)
(`tf.lite.TFLiteConverter`), allowing you to convert your TensorFlow model into
the TensorFlow Lite format and then run an inference:

```python
import numpy as np
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

# Convert to TF Lite format
with tf.Session() as sess:
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Continue to get tensors and so forth, as shown above...
```

For more Python sample code, see [`label_image.py`](
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

Tip: Run `help(tf.lite.Interpreter)` in the Python terminal to get detailed
documentation about the interpreter.


## Write a custom operator

All TensorFlow Lite operators (both custom and builtin) are defined using a
simple pure-C interface that consists of four functions:

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

Refer to `context.h` for details on `TfLiteContext` and `TfLiteNode`. The
former provides error reporting facilities and access to global objects,
including all the tensors. The latter allows implementations to access their
inputs and outputs.

When the interpreter loads a model, it calls `init()` once for each node in the
graph. A given `init()` will be called more than once if the op is used multiple
times in the graph. For custom ops a configuration buffer will be provided,
containing a flexbuffer that maps parameter names to their values. The buffer is
empty for builtin ops because the interpreter has already parsed the op
parameters. Kernel implementations that require state should initialize it here
and transfer ownership to the caller. For each `init()` call, there will be a
corresponding call to `free()`, allowing implementations to dispose of the
buffer they might have allocated in `init()`.

Whenever the input tensors are resized, the interpreter will go through the
graph notifying implementations of the change. This gives them the chance to
resize their internal buffer, check validity of input shapes and types, and
recalculate output shapes. This is all done through `prepare()`, and
implementations can access their state using `node->user_data`.

Finally, each time inference runs, the interpreter traverses the graph calling
`invoke()`, and here too the state is available as `node->user_data`.

Custom ops can be implemented in exactly the same way as builtin ops, by
defined those four functions and a global registration function that usually
looks like this:

```c++
namespace tflite {
namespace ops {
namespace custom {
  TfLiteRegistration* Register_MY_CUSTOM_OP() {
    static TfLiteRegistration r = {my_custom_op::Init,
                                   my_custom_op::Free,
                                   my_custom_op::Prepare,
                                   my_custom_op::Eval};
    return &r;
  }
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

Note that registration is not automatic and an explicit call to
`Register_MY_CUSTOM_OP` should be made somewhere. While the standard
`BuiltinOpResolver` (available from the `:builtin_ops` target) takes care of the
registration of builtins, custom ops will have to be collected in separate
custom libraries.

### Customize the kernel library

Behind the scenes the interpreter will load a library of kernels which will be
assigned to execute each of the operators in the model. While the default
library only contains builtin kernels, it is possible to replace it with a
custom library.

The interpreter uses an `OpResolver` to translate operator codes and names into
actual code:

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddOp(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddOp(const char* op, TfLiteRegistration* registration) = 0;
};
```

Regular usage requires that you use the `BuiltinOpResolver` and write:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

You can optionally register custom ops (before you pass the resolver to the
`InterpreterBuilder`):

```c++
resolver.AddOp("MY_CUSTOM_OP", Register_MY_CUSTOM_OP());
```

If the set of builtin ops is deemed to be too large, a new `OpResolver` could
be code-generated  based on a given subset of ops, possibly only the ones
contained in a given model. This is the equivalent of TensorFlow's selective
registration (and a simple version of it is available in the `tools`
directory).
