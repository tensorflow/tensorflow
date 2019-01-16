# TensorFlow Lite on GPU

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) supports several
hardware accelerators.  This document describes how to use the GPU backend using
the TensorFlow Lite delegate APIs on Android (requires OpenGL ES 3.1 or higher)
and iOS (requires iOS 8 or later).

GPUs are designed to have high throughput for massively parallelizable
workloads.  Thus, they are well-suited for deep neural nets which consists of a
huge number of operators, each working on some input tensor(s) that can be
easily divided into smaller workloads and carried out in parallel, typically
resulting in lower latency.  In the best scenario, inference on the GPU may now
run fast enough and now become suitable for real-time applications if it was not
before.

GPUs do their computation with 16-bit or 32-bit floating point numbers and do
not require quantization for optimal performance unlike the CPUs.  If
quantization of your neural network was not an option due to lower accuracy
caused by lost precision, such concern can be discarded when running deep neural
net models on the GPU.

Another benefit that comes with GPU inference is its power efficiency.  GPUs
carry out the computations in a very efficient and optimized way, so that they
consume less power and generate less heat than when the same task is run on the
CPUs.

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

Using TensorFlow Lite on GPU is achieved via `TfLiteDelegate`.  In Java, you can
specify the GpuDelegate through `Interpreter.Options`.

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

Using TensorFlow Lite on GPU is as simple as getting the GPU delegate via
`NewGpuDelegate()` and then passing it to
`Interpreter::ModifyGraphWithDelegate()` instead of calling
`Interpreter::AllocateTensors()`:

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = NewGpuDelegate(nullptr);  // default config
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// Clean up.
DeleteGpuDelegate(delegate);
```

*IMPORTANT:* When calling `Interpreter::ModifyGraphWithDelegate()` or
`Interpreter::Invoke()`, the caller must have a `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`.
If such `EGLContext` does not exist, the delegate will internally create one,
but then the developer must ensure that `Interpreter::Invoke()` is always called
from the same thread `Interpreter::ModifyGraphWithDelegate()` was called.

## Advanced: Delegate Options for iOS

There are a couple of GPU options that can be set and passed on to
`NewGpuDelegate()`:

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

When option is set to `nullptr` as shown in the Basic Usage, it translates to:

```c++
const GpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = kGpuDelegateOptions::WaitType::Passive,
};
```

While it is convenient to just supply `nullptr`, it is recommended to explicitly
set the options to avoid any unexpected artifacts in case default values are
changed.

## Advanced: Input/Output Buffers

To do computation on the GPU, data must be made available to the GPU which often
translates to performing a memory copy.  It is desirable not to cross the
CPU/GPU memory boundary if possible, as this can take up a significant amount of
time.  Usually, such crossing is inevitable, but in some special cases, one or
the other can be omitted.

If the network's input is an image already loaded in the GPU memory, e.g. a GPU
texture containing the camera feed, it can stay in the GPU memory without ever
entering the CPU memory.  Similarly, if the network's output is in the form of a
renderable image, e.g.
[image style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf),
it can be directly displayed on the screen.

To let users achieve best performance, TensorFlow Lite makes it possible for
them to directly read from/write to the TensorFlow hardware buffer and bypass
avoidable memory copies.

### Android

Assuming the image input is in the GPU memory, it must be first converted to a
OpenGL Shader Storage Buffer Object (SSBO).  One can associate a TfLiteTensor
with user-prepared SSBO with `Interpreter.bindGlBufferToTensor()`.

*IMPORTANT:* `Interpreter.bindGlBufferToTensor()` must be called before
`Interpreter.modifyGraphWithDelegate()`.

```java
// Ensure a valid EGL rendering context.
EGLContext eglContext = eglGetCurrentContext();
if (eglContext.equals(EGL_NO_CONTEXT)) return false;

// Create a SSBO.
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

Similar approach can be applied to the output tensor.  In that case,
`Interpreter.Options.setAllowBufferHandleOutput(true)` should be passed on, to
disable the default copying the network's output from GPU memory to CPU memory.

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

### iOS

Assuming the image input is in the GPU memory, it must be first converted to a
`MTLBuffer` object for Metal.  One can associate a TfLiteTensor with a
user-prepared `MTLBuffer` with `BindMetalBufferToTensor()`.

*IMPORTANT:* `BindMetalBufferToTensor()` must be called before
`Interpreter::ModifyGraphWithDelegate()`.

*IMPORTANT:* By default, the inference output is copied from GPU memory to CPU
memory implicitly by the framework.  This behavior can be turned off by calling
`Interpreter::SetAllowBufferHandleOutput(true)` during initialization.  To copy
the inference output from GPU memory to CPU memory, explicit
`Interpreter::EnsureTensorDataIsReadable()` calls are required for each output
tensor.

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

## Tips and Tricks

* Some operations that are trivial on CPU side may be high cost in GPU land.
  One class of such operation is various forms of reshape operations (including
  `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH`, etc.).  If those ops
  are inserted into the network just for the network architect's logical
  thinking, it is worth removing them for performance.

* On GPU, tensor data is sliced into 4-channels.  Thus, a computation on a
  tensor of shape `[B, H, W, 5]` will perform about the same on a tensor of
  shape `[B, H, W, 8]`, but significantly worse than `[B, H, W, 4]`.

* In that sense, if the camera hardware supports image frames in RGBA, feeding
  that 4-channel input is significantly faster as a memory copy (from 3-channel
  RGB to 4-channel RGBX) can be avoided.

* For best performance, do not hesitate to re-train your classifier with
  mobile-optimized network architecture.  That is a significant part of
  optimization for on-device inference.
