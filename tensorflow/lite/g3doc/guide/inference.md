# TensorFlow Lite inference

The term *inference* refers to the process of executing a TensorFlow Lite model
on-device in order to make predictions based on input data. Inference is the
final step in using the model on-device.

Inference for TensorFlow Lite models is run through an interpreter. The
TensorFlow Lite interpreter is designed to be lean and fast. The interpreter
uses a static graph ordering and a custom (less-dynamic) memory allocator to
ensure minimal load, initialization, and execution latency.

This document outlines the various APIs for the interpreter, along with the
[supported platforms](#supported-platforms).

### Important Concepts

TensorFlow Lite inference on device typically follows the following steps.

1. **Loading a Model**

   The user loads the `.tflite` model into memory which contains the model's
   execution graph.

1. **Transforming Data**
   Input data acquired by the user generally may not match the input data format
   expected by the model. For eg., a user may need to resize an image or change
   the image format to be used by the model.

1. **Running Inference**

   This step involves using the API to execute the model. It involves a few
   steps such as building the interpreter, and allocating tensors as explained
   in detail in [Running a Model](#running_a_model).

1. **Interpreting Output**

   The user retrieves results from model inference and interprets the tensors in
   a meaningful way to be used in the application.

   For example, a model may only return a list of probabilities. It is up to the
   application developer to meaningully map them to relevant categories and
   present it to their user.

### Supported Platforms

TensorFlow inference APIs are provided for most common mobile/embedded platforms
such as Android, iOS and Linux.

#### Android

On Android, TensorFlow Lite inference can be performed using either Java or C++
APIs. The Java APIs provide convenience and can be used directly within your
Android Activity classes. The C++ APIs offer more flexibility and speed, but may
require writing JNI wrappers to move data between Java and C++ layers.

Visit the [Android quickstart](android.md) for a tutorial and example code.

#### iOS

TensorFlow Lite provides native iOS libraries written in
[Swift](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift)
and
[Objective-C](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc).

Visit the [iOS quickstart](ios.md) for a tutorial and example code.

#### Linux
On Linux platforms such as [Raspberry Pi](build_rpi.md), TensorFlow Lite C++
and Python APIs can be used to run inference.


## API Guides

TensorFlow Lite provides programming APIs in C++, Java and Python, with
experimental bindings for several other languages (C, Swift, Objective-C). In
most cases, the API design reflects a preference for performance over ease of
use. TensorFlow Lite is designed for fast inference on small devices so it
should be no surprise that the APIs try to avoid unnecessary copies at the
expense of convenience. Similarly, consistency with TensorFlow APIs was not an
explicit goal and some variance is to be expected.

There is also a [Python API for TensorFlow Lite](../convert/python_api.md).

### Loading a Model

#### C++
The `FlatBufferModel` class encapsulates a model and can be built in a couple of
slightly different ways depending on where the model is stored:

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

```c++
tflite::FlatBufferModel model(path_to_model);
```

Note that if TensorFlow Lite detects the presence of Android's NNAPI it will
automatically try to use shared memory to store the FlatBufferModel.

#### Java

TensorFlow Lite's Java API supports on-device inference and is provided as an
Android Studio Library that allows loading models, feeding inputs, and
retrieving inference outputs.

The `Interpreter` class drives model inference with TensorFlow Lite. In
most of the cases, this is the only class an app developer will need.

The `Interpreter` can be initialized with a model file using the constructor:

```java
public Interpreter(@NotNull File modelFile);
```

or with a `MappedByteBuffer`:

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

In both cases a valid TensorFlow Lite model must be provided or an
`IllegalArgumentException` with be thrown. If a `MappedByteBuffer` is used to
initialize an Interpreter, it should remain unchanged for the whole lifetime of
the `Interpreter`.

### Running a Model {#running_a_model}

#### C++
Running a model involves a few simple steps:

  * Build an `Interpreter` based on an existing `FlatBufferModel`
  * Optionally resize input tensors if the predefined sizes are not desired.
  * Set input tensor values
  * Invoke inference
  * Read output tensor values

The important parts of public interface of the `Interpreter` are provided
below. It should be noted that:

  * Tensors are represented by integers, in order to avoid string comparisons
    (and any fixed dependency on string libraries).
  * An interpreter must not be accessed from concurrent threads.
  * Memory allocation for input and output tensors must be triggered
    by calling AllocateTensors() right after resizing tensors.

In order to run the inference model in TensorFlow Lite, one has to load the
model into a `FlatBufferModel` object which then can be executed by an
`Interpreter`.  The `FlatBufferModel` needs to remain valid for the whole
lifetime of the `Interpreter`, and a single `FlatBufferModel` can be
simultaneously used by more than one `Interpreter`. In concrete terms, the
`FlatBufferModel` object must be created before any `Interpreter` objects that
use it, and must be kept around until they have all been destroyed.

The simplest usage of TensorFlow Lite will look like this:

```c++
tflite::FlatBufferModel model(path_to_model);

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

#### Java

The simplest usage of Tensorflow Lite Java API looks like this:

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

If a model takes only one input and returns only one output, the following will
trigger an inference run:

```java
interpreter.run(input, output);
```

For models with multiple inputs, or multiple outputs, use:

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

where each entry in `inputs` corresponds to an input tensor and
`map_of_indices_to_outputs` maps indices of output tensors to the corresponding
output data. In both cases the tensor indices should correspond to the values
given to the
[TensorFlow Lite Optimized Converter](../convert/cmdline_examples.md) when the
model was created. Be aware that the order of tensors in `input` must match the
order given to the `TensorFlow Lite Optimized Converter`.

The Java API also provides convenient functions for app developers to get the
index of any model input or output using a tensor name:

```java
public int getInputIndex(String tensorName);
public int getOutputIndex(String tensorName);
```

If tensorName is not a valid name in model, an `IllegalArgumentException` will
be thrown.

##### Releasing Resources After Use

An `Interpreter` owns resources. To avoid memory leak, the resources must be
released after use by:

```java
interpreter.close();
```

##### Supported Data Types

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

##### Inputs

Each input should be an array or multi-dimensional array of the supported
primitive types, or a raw `ByteBuffer` of the appropriate size. If the input is
an array or multi-dimensional array, the associated input tensor will be
implicitly resized to the array's dimensions at inference time. If the input is
a ByteBuffer, the caller should first manually resize the associated input
tensor (via `Interpreter.resizeInput()`) before running inference.

When using 'ByteBuffer', prefer using direct byte buffers, as this allows the
`Interpreter` to avoid unnecessary copies. If the `ByteBuffer` is a direct byte
buffer, its order must be `ByteOrder.nativeOrder()`. After it is used for a
model inference, it must remain unchanged until the model inference is finished.

##### Outputs

Each output should be an array or multi-dimensional array of the supported
primitive types, or a ByteBuffer of the appropriate size. Note that some models
have dynamic outputs, where the shape of output tensors can vary depending on
the input. There's no straightforward way of handling this with the existing
Java inference API, but planned extensions will make this possible.


## Writing Custom Operators

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
graph. A given `init()` will be called more than once if the op is used
multiple times in the graph. For custom ops a configuration buffer will be
provided, containing a flexbuffer that maps parameter names to their values.
The buffer is empty for builtin ops because the interpreter has already parsed
the op parameters. Kernel implementation that require state should initialize
it here and transfer ownership to the caller.  For each `init()` call, there
will be a corresponding call to `free()`, allowing implementations to dispose
of the buffer they might have allocated in `init()`.

Whenever the input tensors are resized the interpreter will go through the
graph notifying implementations of the change. This gives them the chance to
resize their internal buffer, check validity of input shapes and types, and
recalculate output shapes. This is all done through `prepare()` and
implementation can access their state using `node->user_data`.

Finally, each time inference runs the interpreter traverses the graph calling
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

### Customizing the kernel library

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

Regular usage will require the developer to use the `BuiltinOpResolver` and
write:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

They can then optionally register custom ops:

```c++
resolver.AddOp("MY_CUSTOM_OP", Register_MY_CUSTOM_OP());
```

before the resolver is passed to the `InterpreterBuilder`.

If the set of builtin ops is deemed to be too large, a new `OpResolver` could
be code-generated  based on a given subset of ops, possibly only the ones
contained in a given model. This is the equivalent of TensorFlow's selective
registration (and a simple version of it is available in the `tools`
directory).
