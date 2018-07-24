book_path: /mobile/_book.yaml
project_path: /mobile/_project.yaml

# TensorFlow Lite APIs

TensorFlow Lite provides programming APIs in C++ and Java, and in both cases
the API design reflects a preference for performance over ease of use.
TensorFlow Lite is designed for fast inference on small devices so it should be
no surprise that the APIs try to avoid unnecessary copies at the expense of
convenience.  Similarly, consistency with TensorFlow APIs was not an explicit
goal and some variance is to be expected.

There is also a Python API for TensorFlow Lite described
[here](../toco/g3doc/python_api.md#interpreter).

## C++

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
### Data Alignment

TensorFlow Lite data is usually aligned to 32-bit boundaries. It is recommended
that all data provided to TensorFlow Lite be aligned that way.

### Error Reporting

In many places TensorFlow Lite returns status information through
`TfLiteStatus` objects:

```c++
typedef enum {
  kTfLiteOk = 0,
  kTfLiteError = 1
} TfLiteStatus;

```

Failures can be easily verified with:

```c++
if (status != kTfLiteOk) {
  // ... error handling here ...
}
```

In order to obtain detailed error information an ErrorReporter must be
provided:

```c++
class ErrorReporter {
  virtual int Report(const char* format, va_list args) = 0;
};
```

The `DefaultErrorReporter` takes care of reporting to `stderr`.

### Loading a Model

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

Note that if TensorFlow Lite detects the presence of Android's NNAPI it will
automatically try to use shared memory to store the FlatBufferModel.

### Running a Model

Running a model involves a few simple steps:

  * Build an `Interpreter` based on an existing `FlatBufferModel`
  * Optionally resize input tensors if the predefined sizes are not desired.
  * Set input tensor values
  * Invoke inference
  * Read output tensor values

The important parts of public interface of the `Interpreter` are provided
below.  It should be noted that:

  * Tensors are represented by integers, in order to avoid string comparisons
    (and any fixed dependency on string libraries).
  * An interpreter must not be accessed from concurrent threads
  * Memory allocation for input and output tensors must be triggered
    by calling AllocateTensors() right after resizing tensors.

```c++
class Interpreter {
  Interpreter(ErrorReporter* error_reporter);

  // Read only access to list of inputs.
  const std::vector<int>& inputs() const;

  // Read only access to list of outputs.
  const std::vector<int>& outputs() const;

  // Change the dimensionality of a given tensor.
  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  // Returns status of success or failure.
  TfLiteStatus AllocateTensors();

  // Return a pointer into the data of a given input tensor.
  template <class T>
  T* typed_input_tensor(int index) {
    return typed_tensor<T>(inputs_[index]);
  }

  // Return a pointer into the data of a given output tensor.
  template <class T>
  T* typed_output_tensor(int index) {
    return typed_tensor<T>(outputs_[index]);
  }

  // Execute the model, populating output tensors.
  TfLiteStatus Invoke();
};
```

### Writing Custom Operators

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

When the interpreter loads a model, it calls init() once for each node in the
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
`:builtin_ops` takes care of the registration of builtins, custom ops will have
to be collected in separated custom libraries.

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

The regular usage will require the developer to use the `BuiltinOpResolver` and
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

## Java

TensorFlow Lite's Java API supports on-device inference and is provided as an
Android Studio Library that allows loading models, feeding inputs, and
retrieving inference outputs.

The simplest usage of Tensorflow Lite Java API looks like this:

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

### Loading a Model

The `Interpreter.java` class drives model inference with TensorFlow Lite. In
most of the cases, this is the only class an app developer will need.

#### Initializing an `Interpreter` With a Model File

The `Interpreter` can be initialized with a model file using the constructor:

```java
public Interpreter(@NotNull File modelFile);
```

or with a `MappedByteBuffer`:

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

In both cases a valid TensorFlow Lite must be provided or an
`IllegalArgumentException` with be thrown. If a `MappedByteBuffer` is used to
initialize an Interpreter, it should remain unchanged for the whole lifetime of
the `Interpreter`.

### Running a Model

#### Supported Data Types

To use TensorFlow Lite, the data types of the input and output tensors must be
one of the following primitive types:

*   `float`
*   `int`
*   `long`
*   `byte`

If other data types, including boxed types like `Integer` and `Float`, are used,
an `IllegalArgumentException` will be thrown.

#### Inputs

Each input should be an array, a multi-dimensional array, or a `ByteBuffer` of
the supported primitive types.

The use of `ByteBuffer` is preferred since it allows the `Interpreter` to avoid
unnecessary copies. Each `ByteBuffer` needs to be a direct byte buffer, and its
order must be `ByteOrder.nativeOrder()`. After it is used for a model inference,
it must remain unchanged until the model inference is finished.

#### Outputs

Each output should be an array, or a multi-dimensional array of the supported
primitive types.

#### Running Model Inference

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
`map_of_indices_to_outputs` maps indices of output tensors to the
corresponding output data. In both cases the tensor indices should correspond to
the values given to the `TensorFlow Lite Optimized Converter` when the model was
created. Be aware that the order of tensors in `input` must match the order
given to the `TensorFlow Lite Optimized Converter`.

The Java API also provides convenient functions for app developers to get the
index of any model input or output using a tensor name:

```java
public int getInputIndex(String tensorName);
public int getOutputIndex(String tensorName);
```

If tensorName is not a valid name in model, an `IllegalArgumentException` will
be thrown.

### Releasing Resources After Use

An `Interpreter` owns resources. To avoid memory leak, the resources must be
released after use by:

```java
interpreter.close();
```
