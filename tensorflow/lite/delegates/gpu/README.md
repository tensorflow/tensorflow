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
* `EXP v1`
* `FULLY_CONNECTED v1`
* `LOGISTIC v1`
* `LSTM v2 (Basic LSTM only)`
* `MAX_POOL_2D v1`
* `MAXIMUM v1`
* `MINIMUM v1`
* `MUL v1`
* `PAD v1`
* `PRELU v1`
* `RELU v1`
* `RELU6 v1`
* `RESHAPE v1`
* `RESIZE_BILINEAR v1-3`
* `SOFTMAX v1`
* `STRIDED_SLICE v1`
* `SUB v1`
* `TRANSPOSE_CONV v1`

## Basic Usage

**Note:** Following section describes the example usage for Android GPU delegate
with C++. For other languages and platforms, please see
[the documentation](https://www.tensorflow.org/lite/performance/gpu).

Using TFLite on GPU is as simple as getting the GPU delegate via
`TfLiteGpuDelegateV2Create()` and then passing it to
`InterpreterBuilder::AddDelegate()`:

```c++
////////
// Set up InterpreterBuilder.
auto model = FlatBufferModel::BuildFromFile(model_path);
ops::builtin::BuiltinOpResolver op_resolver;
InterpreterBuilder interpreter_builder(*model, op_resolver);

////////
// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
interpreter_builder.AddDelegate(delegate);

////////
// Set up Interpreter.
std::unique_ptr<Interpreter> interpreter;
if (interpreter_builder(&interpreter) != kTfLiteOk) return;

////////
// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

////////
// Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

*IMPORTANT:* When calling `Interpreter::ModifyGraphWithDelegate()` or
`InterpreterBuilder::operator()` or
`Interpreter::Invoke()`, the caller must have a `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`.
If such `EGLContext` does not exist, the delegate will internally create one,
but then the developer must ensure that `Interpreter::Invoke()` is always called
from the same thread `InterpreterBuilder::operator()` or
`Interpreter::ModifyGraphWithDelegate()` was called.

## Building and Runtime

TFLite GPU backend uses OpenGL ES 3.1 compute shaders or OpenCL.

```sh
bazel build --config android_arm64 //path/to/your:project
```

Metal shaders are used for iOS, which were introduced with iOS 8.  Thus,
compilation flags should look like:

```sh
bazel build --config ios_fat //path/to/your:project
```

## Advanced Usage: Delegate Options

There are GPU options that can be set and passed on to
`TfLiteGpuDelegateV2Create()`. When option is set to `nullptr` as shown in the
Basic Usage, it translates to:

```c++
const TfLiteGpuDelegateOptionsV2 kDefaultOptions =
    TfLiteGpuDelegateOptionsV2Default();
```

Similar for `TFLGpuDelegateCreate()`:

```c++
const TFLGpuDelegateOptions kDefaultOptions = {
  .allow_precision_loss = false,
  .wait_type = TFLGpuDelegateWaitTypePassive,
  .enable_quantization = false,
};
```

While it is convenient to just supply `nullptr`, it is recommended to explicitly
set the options to avoid any unexpected artifacts in case default values are
changed.

*IMPORTANT:* Note that the default option may not be the fastest. For faster
execution, you may want to set `allow_precision_loss` to `true` so that the GPU
performs FP16 calculation internally, and set `wait_type` to
`TFLGpuDelegateWaitTypeAggressive` to avoid GPU sleep mode.

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

## Publication

*   [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/abs/1907.01989)
    *   Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk, Mogan
        Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, and Matthias
        Grundmann
    *   CVPR Workshop
        [Efficient Deep Learning for Computer Vision (ECV2019)](https://sites.google.com/corp/view/ecv2019)
