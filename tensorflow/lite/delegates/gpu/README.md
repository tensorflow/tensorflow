# TFLite on GPU

TensorFlow Lite (TFLite) supports several hardware accelerators.  This document
describes how to use the GPU backend using the TFLite delegate APIs on Android
and iOS.

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

TFLite on GPU supports the following ops in 16-bit and 32-bit float precision:

* `ADD v1`
* `AVERAGE_POOL_2D v1`
* `CONCATENATION v1`
* `CONV_2D v1`
* `DEPTHWISE_CONV_2D v1-2`
* `FULLY_CONNECTED v1`
* `LOGISTIC v1`
* `LSTM v2 (Basic LSTM only)`
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

Using TFLite on GPU is as simple as getting the GPU delegate via
`TfLiteGpuDelegateCreate()` and then passing it to
`Interpreter::ModifyGraphWithDelegate()` instead of calling
`Interpreter::AllocateTensors()`:

```c++
////////
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

////////
// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateCreate(/*options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return;

////////
// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

////////
// Clean up.
TfLiteGpuDelegateDelete(delegate);
```

*IMPORTANT:* When calling `Interpreter::ModifyGraphWithDelegate()` or
`Interpreter::Invoke()`, the caller must have a `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`.
If such `EGLContext` does not exist, the delegate will internally create one,
but then the developer must ensure that `Interpreter::Invoke()` is always called
from the same thread `Interpreter::ModifyGraphWithDelegate()` was called.

## Building and Runtime

TFLite GPU backend uses OpenGL compute shaders and thus requires OpenGL ES 3.1
or higher.

```sh
bazel build --config android_arm64 //path/to/your:project
```

Metal shaders are used for iOS, which were introduced with iOS 8.  Thus,
compilation flags should look like:

```sh
bazel build --config ios_arm64 //path/to/your:project
```

## Advanced Usage: Delegate Options

There are GPU options that can be set and passed on to
`TfLiteGpuDelegateCreate()`. When option is set to `nullptr` as shown in the
Basic Usage, it translates to:

```c++
const TfLiteGpuDelegateOptions kDefaultOptions = {
  .metadata = nullptr,
  .compile_options = {
    .precision_loss_allowed = 0,  // false
    .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
    .dynamic_batch_enabled = 0,  // false
  },
};
```

Similar for `NewTfLiteMetalDelgate()`:

```c++
const TfLiteMetalDelegateOptions kDefaultOptions = {
  .precision_loss_allowed = 0,  // false
  .wait_type = TFLITE_METAL_WAIT_TYPE_SLEEP,
};
```

While it is convenient to just supply `nullptr`, it is recommended to explicitly
set the options to avoid any unexpected artifacts in case default values are
changed.

*IMPORTANT:* Note that the default option does not allow precision loss, and
thus may not be the fastest.  For faster execution, you may want to set
`precision_loss_allowed` to `1` for FP16 execution.

## Advanced Usage: Input/Output Buffers (C++)

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

To let users achieve best performance, TFLite makes it possible for them to
directly read from/write to the delegate's hardware buffer and bypass avoidable
memory copies.

Assuming the camera input is in the GPU memory as `GL_TEXTURE_2D`, it must be
first converted to a shader storage buffer object (SSBO) for OpenGL or to a
`MTLBuffer` object for Metal. One can associate a TfLiteTensor with a
user-prepared SSBO or `MTLBuffer` with `TfLiteGpuDelegateBindBufferToTensor()`
or `TfLiteMetalDelegateBindBufferToTensor()`, respectively.

*IMPORTANT:* These must be called before
`Interpreter::ModifyGraphWithDelegate()`.

*IMPORTANT:* By default, the inference output is copied from GPU memory to CPU
memory implicitly by the framework.  This behavior can be turned off by calling
`Interpreter::SetAllowBufferHandleOutput(true)` during initialization.  To copy
the inference output from GPU memory to CPU memory, explicit
`Interpreter::EnsureTensorDataIsReadable()` calls are required for each output
tensor.

```c++
////////
// Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateCreate(nullptr);
interpreter->SetAllowBufferHandleOutput(true);  // disable default gpu->cpu copy
#if defined(__ANDROID__)
if (TfLiteGpuDelegateBindBufferToTensor(delegate, user_provided_input_buffer, interpreter->inputs()[0]) != kTfLiteOk) return;
if (TfLiteGpuDelegateBindBufferToTensor(delegate, user_provided_output_buffer, interpreter->outputs()[0]) != kTfLiteOk) return;
#elif defined(__APPLE__)
if (TfLiteMetalDelegateBindBufferToTensor(delegate, user_provided_input_buffer, interpreter->inputs()[0]) != kTfLiteOk) return;
if (TfLiteMetalDelegateBindBufferToTensor(delegate, user_provided_output_buffer, interpreter->outputs()[0]) != kTfLiteOk) return;
#endif
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return;

////////
// Run inference.
if (interpreter->Invoke() != kTfLiteOk) return;
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

* For performance [best practices](https://www.tensorflow.org/lite/performance/best_practices), do not hesitate to re-train your classifier with
  mobile-optimized network architecture.  That is a significant part of
  optimization for on-device inference.
