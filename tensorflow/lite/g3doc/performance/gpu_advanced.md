# TensorFlow Lite on GPU

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) supports several
hardware accelerators.  This document describes how to use the GPU backend using
the TensorFlow Lite delegate APIs on Android (requires OpenGL ES 3.1 or higher)
and iOS (requires iOS 8 or later).

## Benefits of GPU Acceleration

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

### Energy Efficiency

Another benefit that comes with GPU inference is its power efficiency. A GPU
carries out computations in a very efficient and optimized way, consuming less
power and generating less heat than the same task run on a CPU.

## Supported Ops

TensorFlow Lite on GPU supports the following ops in 16-bit and 32-bit float
precision:

* `ADD v1`
* `AVERAGE_POOL_2D v1`
* `CONCATENATION v1`
* `CONV_2D v1`
* `DEPTHWISE_CONV_2D v1-2`
* `FULLY_CONNECTED v1`
* `LOGISTIC v1`
* `MAX_POOL_2D v1`
* `MUL v1`
* `PAD v1`
* `PRELU v1`
* `RELU v1`
* `RELU6 v1`
* `RESHAPE v1`
* `RESIZE_BILINEAR v1`
* `SOFTMAX v1`
* `STRIDED_SLICE v1`
* `SUB v1`
* `TRANSPOSE_CONV v1`

## Basic Usage

### Android

Run TensorFlow Lite on GPU with `TfLiteDelegate`. In Java, you can specify the
GpuDelegate through `Interpreter.Options`.

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

### iOS

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

## Advanced Usage

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

### Input/Output Buffers

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

#### Android

Assuming the image input is in the GPU memory, it must first be converted to an
OpenGL Shader Storage Buffer Object (SSBO). You can associate a TfLiteTensor to
a user-prepared SSBO with `Interpreter.bindGlBufferToTensor()`. Note that
`Interpreter.bindGlBufferToTensor()` must be called before
`Interpreter.modifyGraphWithDelegate()`.

```java
// Ensure a valid EGL rendering context.
EGLContext eglContext = eglGetCurrentContext();
if (eglContext.equals(EGL_NO_CONTEXT)) return false;

// Create an SSBO.
int[] id = new int[1];
glGenBuffers(id.length, id, 0);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
glBufferData(GL_SHADER_STORAGE_BUFFER, inputSize, null, GL_STREAM_COPY);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind
int inputSsboId = id[0];

// Create interpreter.
Interpreter interpreter = new Interpreter(tfliteModel);
Tensor inputTensor = interpreter.getInputTensor(0);
GpuDelegate gpuDelegate = new GpuDelegate();
// The buffer must be bound before the delegate is installed.
gpuDelegate.bindGlBufferToTensor(inputTensor, inputSsboId);
interpreter.modifyGraphWithDelegate(gpuDelegate);

// Run inference; the null input argument indicates use of the bound buffer for input.
fillSsboWithCameraImageTexture(inputSsboId);
float[] outputArray = new float[outputSize];
interpreter.runInference(null, outputArray);
```

A similar approach can be applied to the output tensor. In that case,
`Interpreter.Options.setAllowBufferHandleOutput(true)` should be passed on, to
disable the default copying of the network's output from GPU memory to CPU
memory.

```java
// Ensure a valid EGL rendering context.
EGLContext eglContext = eglGetCurrentContext();
if (eglContext.equals(EGL_NO_CONTEXT)) return false;

// Create a SSBO.
int[] id = new int[1];
glGenBuffers(id.length, id, 0);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
glBufferData(GL_SHADER_STORAGE_BUFFER, outputSize, null, GL_STREAM_COPY);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind
int outputSsboId = id[0];

// Create interpreter.
Interpreter.Options options = (new Interpreter.Options()).setAllowBufferHandleOutput(true);
Interpreter interpreter = new Interpreter(tfliteModel, options);
Tensor outputTensor = interpreter.getOutputTensor(0);
GpuDelegate gpuDelegate = new GpuDelegate();
// The buffer must be bound before the delegate is installed.
gpuDelegate.bindGlBufferToTensor(outputTensor, outputSsboId);
interpreter.modifyGraphWithDelegate(gpuDelegate);

// Run inference; the null output argument indicates use of the bound buffer for output.
ByteBuffer input = getCameraImageByteBuffer();
interpreter.runInference(input, null);
renderOutputSsbo(outputSsboId);
```

#### iOS

Assuming the image input is in GPU memory, it must first be converted to a
`MTLBuffer` object for Metal. You can associate a TfLiteTensor to a
user-prepared `MTLBuffer` with `BindMetalBufferToTensor()`. Note that
`BindMetalBufferToTensor()` must be called before
`Interpreter::ModifyGraphWithDelegate()`. Additionally, the inference output is,
by default, copied from GPU memory to CPU memory. This behavior can be turned
off by calling `Interpreter::SetAllowBufferHandleOutput(true)` during
initialization.

```c++
// Prepare GPU delegate.
auto* delegate = NewGpuDelegate(nullptr);
interpreter->SetAllowBufferHandleOutput(true);  // disable default gpu->cpu copy
if (!BindMetalBufferToTensor(delegate, interpreter->inputs()[0], user_provided_input_buffer)) return false;
if (!BindMetalBufferToTensor(delegate, interpreter->outputs()[0], user_provided_output_buffer)) return false;
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
