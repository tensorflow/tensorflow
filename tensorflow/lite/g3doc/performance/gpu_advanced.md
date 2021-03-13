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

There are two ways to invoke model acceleration in Android depending on if you
are using
[Android Studio ML Model Binding](../inference_with_metadata/codegen#acceleration)
or TensorFlow Lite Interpreter.

### Android via TensorFlow Lite Interpreter

Add the `tensorflow-lite-gpu` package alongside the existing `tensorflow-lite`
package in the existing `dependencies` block.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

Then run TensorFlow Lite on GPU with `TfLiteDelegate`. In Java, you can specify
the `GpuDelegate` through `Interpreter.Options`.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p><pre class="prettyprint lang-kotlin">
    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
        if(compatList.isDelegateSupportedOnThisDevice){
            // if the device has a supported GPU, add the GPU delegate
            val delegateOptions = compatList.bestOptionsForThisDevice
            this.addDelegate(GpuDelegate(delegateOptions))
        } else {
            // if the GPU is not supported, run on 4 threads
            this.setNumThreads(4)
        }
    }

    val interpreter = Interpreter(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
    import org.tensorflow.lite.Interpreter;
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
    } else {
        // if the GPU is not supported, run on 4 threads
        options.setNumThreads(4);
    }

    Interpreter interpreter = new Interpreter(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre></p>
    </section>
  </devsite-selector>
</div>

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
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

Note: When calling `Interpreter::ModifyGraphWithDelegate()` or
`Interpreter::Invoke()`, the caller must have an `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`. If
an `EGLContext` does not exist, the delegate will internally create one, but
then the developer must ensure that `Interpreter::Invoke()` is always called
from the same thread in which `Interpreter::ModifyGraphWithDelegate()` was
called.

### iOS (C++)

Note: For Swift/Objective-C/C use cases, please refer to
[GPU delegate guide](gpu#ios)

Note: This is only available when you are using bazel or build TensorFlow Lite
by yourself. C++ API can't be used with CocoaPods.

To use TensorFlow Lite on GPU, get the GPU delegate via `TFLGpuDelegateCreate()`
and then pass it to `Interpreter::ModifyGraphWithDelegate()` (instead of calling
`Interpreter::AllocateTensors()`).

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.

auto* delegate = TFLGpuDelegateCreate(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// Clean up.
TFLGpuDelegateDelete(delegate);
```

## Advanced usage

### Delegate Options for iOS

Constructor for GPU delegate accepts a `struct` of options.
([Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift),
[Objective-C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h),
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h))

Passing `nullptr` (C API) or nothing (Objective-C and Swift API) to the
initializer sets the default options (which are explicated in the Basic Usage
example above).

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p><pre class="prettyprint lang-swift">
    // THIS:
    var options = MetalDelegate.Options()
    options.isPrecisionLossAllowed = false
    options.waitType = .passive
    options.isQuantizationEnabled = true
    let delegate = MetalDelegate(options: options)

    // IS THE SAME AS THIS:
    let delegate = MetalDelegate()
      </pre></p>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p><pre class="prettyprint lang-objc">
    // THIS:
    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.precisionLossAllowed = false;
    options.waitType = TFLMetalDelegateThreadWaitTypePassive;
    options.quantizationEnabled = true;

    TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] initWithOptions:options];

    // IS THE SAME AS THIS:
    TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] init];
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-c">
    // THIS:
    const TFLGpuDelegateOptions options = {
      .allow_precision_loss = false,
      .wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive,
      .enable_quantization = true,
    };

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);

    // IS THE SAME AS THIS:
    TfLiteDelegate* delegate = TFLGpuDelegateCreate(nullptr);
      </pre></p>
    </section>
  </devsite-selector>
</div>

While it is convenient to use `nullptr` or default constructors, we recommend
that you explicitly set the options, to avoid any unexpected behavior if default
values are changed in the future.

### Running quantized models on GPU

This section explains how the GPU delegate accelerates 8-bit quantized models.
This includes all flavors of quantization, including:

*   Models trained with
    [Quantization-aware training](https://www.tensorflow.org/lite/convert/quantization)
*   [Post-training dynamic-range quantization](https://www.tensorflow.org/lite/performance/post_training_quant)
*   [Post-training full-integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

To optimize performance, use models that have floating-point input & output
tensors.

#### How does this work?

Since the GPU backend only supports floating-point execution, we run quantized
models by giving it a ‘floating-point view’ of the original model. At a
high-level, this entails the following steps:

*   *Constant tensors* (such as weights/biases) are dequantized once into the
    GPU memory. This happens when the delegate is applied to the TFLite
    Interpreter.

*   *Inputs and outputs* to the GPU program, if 8-bit quantized, are dequantized
    and quantized (respectively) for each inference. This is done on the CPU
    using TFLite’s optimized kernels.

*   The GPU program is modified to mimic quantized behavior by inserting
    *quantization simulators* between operations. This is necessary for models
    where ops expect activations to follow bounds learnt during quantization.

This feature can be enabled using delegate options as follows:

#### Android

Android APIs support quantized models by default. To disable, do the following:

**C++ API**

```c++
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
```

**Java API**

```java
GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
```

#### iOS

iOS APIs support quantized models by default. To disable, do the following:

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p><pre class="prettyprint lang-swift">
    var options = MetalDelegate.Options()
    options.isQuantizationEnabled = false
    let delegate = MetalDelegate(options: options)
      </pre></p>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p><pre class="prettyprint lang-objc">
    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.quantizationEnabled = false;
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-c">
    TFLGpuDelegateOptions options = TFLGpuDelegateOptionsDefault();
    options.enable_quantization = false;

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);
      </pre></p>
    </section>
  </devsite-selector>
</div>

### Input/Output Buffers (iOS, C++ API only)

Note: This is only available when you are using bazel or build TensorFlow Lite
by yourself. C++ API can't be used with CocoaPods.

To do computation on the GPU, data must be made available to the GPU. This often
requires performing a memory copy. It is desirable not to cross the CPU/GPU
memory boundary if possible, as this can take up a significant amount of time.
Usually, such crossing is inevitable, but in some special cases, one or the
other can be omitted.

If the network's input is an image already loaded in the GPU memory (for
example, a GPU texture containing the camera feed) it can stay in the GPU memory
without ever entering the CPU memory. Similarly, if the network's output is in
the form of a renderable image (for example,
[image style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf))
it can be directly displayed on the screen.

To achieve best performance, TensorFlow Lite makes it possible for users to
directly read from and write to the TensorFlow hardware buffer and bypass
avoidable memory copies.

Assuming the image input is in GPU memory, it must first be converted to a
`MTLBuffer` object for Metal. You can associate a TfLiteTensor to a
user-prepared `MTLBuffer` with `TFLGpuDelegateBindMetalBufferToTensor()`. Note
that `TFLGpuDelegateBindMetalBufferToTensor()` must be called after
`Interpreter::ModifyGraphWithDelegate()`. Additionally, the inference output is,
by default, copied from GPU memory to CPU memory. This behavior can be turned
off by calling `Interpreter::SetAllowBufferHandleOutput(true)` during
initialization.

```c++
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate_internal.h"

// ...

// Prepare GPU delegate.
auto* delegate = TFLGpuDelegateCreate(nullptr);

if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

interpreter->SetAllowBufferHandleOutput(true);  // disable default gpu->cpu copy
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter->inputs()[0], user_provided_input_buffer)) {
  return false;
}
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter->outputs()[0], user_provided_output_buffer)) {
  return false;
}

// Run inference.
if (interpreter->Invoke() != kTfLiteOk) return false;
```

Note: Once the default behavior is turned off, copying the inference output from
GPU memory to CPU memory requires an explicit call to
`Interpreter::EnsureTensorDataIsReadable()` for each output tensor.

Note: This also works for quantized models, but you still need to a **float32
sized buffer with float32 data**, because the buffer will be bound to the
internal dequantized buffer.

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
