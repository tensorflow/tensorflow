# TensorFlow Lite on GPU

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) supports several
hardware accelerators. This document describes how to use the GPU backend using
the TensorFlow Lite delegate APIs on Android (requires OpenCL or OpenGL ES 3.1
and higher) and iOS (requires iOS 8 or later).

## Benefits of GPU acceleration

### Speed

GPUs are designed to have high throughput for massively parallelizable
workloads. Thus, they are well-suited for deep neural nets, which consist of a
huge number of operators, each working on some input tensor(s) that can be
easily divided into smaller workloads and carried out in parallel. This
parallelism typically results in lower latency. In the best scenario, inference
on the GPU may run fast enough to become suitable for real-time applications
that were not previously possible.

### Accuracy

GPUs do their computation with 16-bit or 32-bit floating point numbers and
(unlike the CPUs) do not require quantization for optimal performance. If
decreased accuracy made quantization untenable for your models, running your
neural network on a GPU may eliminate this concern.

### Energy efficiency

Another benefit that comes with GPU inference is its power efficiency. A GPU
carries out computations in a very efficient and optimized way, consuming less
power and generating less heat than the same task run on a CPU.

## Supported ops

TensorFlow Lite on GPU supports the following ops in 16-bit and 32-bit float
precision:

*   `ADD`
*   `AVERAGE_POOL_2D`
*   `CONCATENATION`
*   `CONV_2D`
*   `DEPTHWISE_CONV_2D v1-2`
*   `EXP`
*   `FULLY_CONNECTED`
*   `LOGISTIC`
*   `LSTM v2 (Basic LSTM only)`
*   `MAX_POOL_2D`
*   `MAXIMUM`
*   `MINIMUM`
*   `MUL`
*   `PAD`
*   `PRELU`
*   `RELU`
*   `RELU6`
*   `RESHAPE`
*   `RESIZE_BILINEAR v1-3`
*   `SOFTMAX`
*   `STRIDED_SLICE`
*   `SUB`
*   `TRANSPOSE_CONV`

By default, all ops are only supported at version 1. Enabling the
[experimental quantization support](gpu_advanced.md#running-quantized-models-experimental-android-only)
allows the appropriate versions; for example, ADD v2.

## Basic usage

### Android (Java)

Run TensorFlow Lite on GPU with `TfLiteDelegate`. In Java, you can specify the
`GpuDelegate` through `Interpreter.Options`.

```java
// NEW: Prepare GPU delegate.
GpuDelegate delegate = new GpuDelegate();
Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);

// Set up interpreter.
Interpreter interpreter = new Interpreter(model, options);

// Run inference.
writeToInputTensor(inputTensor);
interpreter.run(inputTensor, outputTensor);
readFromOutputTensor(outputTensor);

// Clean up.
delegate.close();
```

### Android (C/C++)

For C/C++ usage of TensorFlow Lite GPU on Android, the GPU delegate can be
created with `TfLiteGpuDelegateV2Create()` and destroyed with
`TfLiteGpuDelegateV2Delete()`.

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

Take a look at `TfLiteGpuDelegateOptionsV2` to create a delegate instance with
custom options. You can initialize the default options with
`TfLiteGpuDelegateOptionsV2Default()` and then modify them as necessary.

TFLite GPU for Android C/C++ uses the [Bazel](https://bazel.io) build system.
The delegate can be built, for example, using the following command:

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:gl_delegate                  # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so  # for dynamic library
```

### iOS (ObjC++)

To use TensorFlow Lite on GPU, get the GPU delegate via `NewGpuDelegate()` and
then pass it to `Interpreter::ModifyGraphWithDelegate()` (instead of calling
`Interpreter::AllocateTensors()`).

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.

const GpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = kGpuDelegateOptions::WaitType::Passive,
};

auto* delegate = NewGpuDelegate(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// Clean up.
DeleteGpuDelegate(delegate);
```

Note: When calling `Interpreter::ModifyGraphWithDelegate()` or
`Interpreter::Invoke()`, the caller must have an `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`. If
an `EGLContext` does not exist, the delegate will internally create one, but
then the developer must ensure that `Interpreter::Invoke()` is always called
from the same thread in which `Interpreter::ModifyGraphWithDelegate()` was
called.

## Advanced usage

### Running quantized models (Experimental, Android only)

The GPU delegate already supports
[float16 quantized](https://www.tensorflow.org/lite/performance/post_training_float16_quant)
models. There is experimental support on Android to run 8-bit quantized as well.
This includes all flavors of quantization, including:

*   Models trained with
    [Quantization-aware training](https://www.tensorflow.org/lite/convert/quantization)
*   [Post-training dynamic-range quantization](https://www.tensorflow.org/lite/performance/post_training_quant)
*   [Post-training full-integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

To optimize performance, use models that have floating-point input & output
tensors.

This feature can be enabled using delegate options as follows:

**C++ API**

```c++
// NEW: Prepare custom options with feature enabled.
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
```

**Java API**

```java
// NEW: Prepare GPU delegate with feature turned on.
GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(true));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
```

### Delegate Options for iOS

`NewGpuDelegate()` accepts a `struct` of options.

```c++
struct GpuDelegateOptions {
  // Allows to quantify tensors, downcast values, process in float16 etc.
  bool allow_precision_loss;

  enum class WaitType {
    // waitUntilCompleted
    kPassive,
    // Minimize latency. It uses active spinning instead of mutex and consumes
    // additional CPU resources.
    kActive,
    // Useful when the output is used with GPU pipeline then or if external
    // command encoder is set
    kDoNotWait,
  };
  WaitType wait_type;
};
```

Passing `nullptr` into `NewGpuDelegate()` sets the default options (which are
explicated in the Basic Usage example above).

```c++

// THIS:
const GpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = kGpuDelegateOptions::WaitType::Passive,
};

auto* delegate = NewGpuDelegate(options);

// IS THE SAME AS THIS:
auto* delegate = NewGpuDelegate(nullptr);

```

While it is convenient to use `nullptr`, we recommend that you explicitly set
the options, to avoid any unexpected behavior if default values are changed in
the future.

### Input/Output Buffers (iOS only)

To do computation on the GPU, data must be made available to the GPU. This often
requires performing a memory copy. It is desirable not to cross the CPU/GPU
memory boundary if possible, as this can take up a significant amount of time.
Usually, such crossing is inevitable, but in some special cases, one or the
other can be omitted.

If the network's input is an image already loaded in the GPU memory (for
example, a GPU texture containing the camera feed) it can stay in the GPU memory
without ever entering the CPU memory. Similarly, if the network's output is in
the form of a renderable image (for example,
[image style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)_)
it can be directly displayed on the screen.

To achieve best performance, TensorFlow Lite makes it possible for users to
directly read from and write to the TensorFlow hardware buffer and bypass
avoidable memory copies.

Assuming the image input is in GPU memory, it must first be converted to a
`MTLBuffer` object for Metal. You can associate a TfLiteTensor to a
user-prepared `MTLBuffer` with `TFLGpuDelegateBindMetalBufferToTensor()`. Note
that `TFLGpuDelegateBindMetalBufferToTensor()` must be called before
`Interpreter::ModifyGraphWithDelegate()`. Additionally, the inference output is,
by default, copied from GPU memory to CPU memory. This behavior can be turned
off by calling `Interpreter::SetAllowBufferHandleOutput(true)` during
initialization.

```c++
// Prepare GPU delegate.
auto* delegate = NewGpuDelegate(nullptr);
interpreter->SetAllowBufferHandleOutput(true);  // disable default gpu->cpu copy
if (!TFLGpuDelegateBindMetalBufferToTensor(delegate, interpreter->inputs()[0], user_provided_input_buffer)) return false;
if (!TFLGpuDelegateBindMetalBufferToTensor(delegate, interpreter->outputs()[0], user_provided_output_buffer)) return false;
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
if (interpreter->Invoke() != kTfLiteOk) return false;
```

Note: Once the default behavior is turned off, copying the inference output from
GPU memory to CPU memory requires an explicit call to
`Interpreter::EnsureTensorDataIsReadable()` for each output tensor.

## Tips and Tricks

*   Some operations that are trivial on the CPU may be high cost on a GPU. One
    class of such operation includes various forms of reshape operations
    (including `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH`, and similar
    operation). If these operations are not required (for example, they were
    inserted to help the network architect reason about the system but do not
    otherwise affect output), it is worth removing them for performance.

*   On a GPU, tensor data is sliced into 4-channels. Thus, a computation on a
    tensor of shape `[B, H, W, 5]` will perform about the same on a tensor of
    shape `[B, H, W, 8]`, but significantly worse than `[B, H, W, 4]`.

    *   For example, if the camera hardware supports image frames in RGBA,
        feeding that 4-channel input is significantly faster, because a memory
        copy (from 3-channel RGB to 4-channel RGBX) can be avoided.

*   For best performance, do not hesitate to re-train your classifier with
    mobile-optimized network architecture. That is a significant part of
    optimization for on-device inference.
