# XNNPACK backend for TensorFlow Lite

XNNPACK is a highly optimized library of floating-point neural network
inference operators for ARM, x86, and WebAssembly architectures in Android, iOS,
Windows, Linux, macOS, and Emscripten environments. This document describes how
to use the XNNPACK library as an inference engine for TensorFlow Lite.

## Using XNNPACK engine with TensorFlow Lite interpreter

XNNPACK integrates with TensorFlow Lite interpreter through the delegation
mechanism. There are three methods to enable XNNPACK engine in TensorFlow Lite.

### Enable XNNPACK via Bazel build flags (recommended)

When building TensorFlow Lite with Bazel, add
`--define tflite_with_xnnpack=true`, and the TensorFlow Lite interpreter will
use XNNPACK engine by default.

The exact command depends on the target platform, e.g. for Android AAR you'd use

```
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --define tflite_with_xnnpack=true \
  //tensorflow/lite/java:tensorflow-lite
```

### Enable XNNPACK via additional dependency

Another way to enable XNNPACK is to build and link the
`//tensorflow/lite:tflite_with_xnnpack` target into your application alongside
the TensorFlow Lite framework.

This method works on platforms which support POSIX-style weak symbols (Android,
iOS, Linux, Mac, but **NOT** Windows).

### Enable XNNPACK via low-level delegate API (not recommended)

While it is possible to use low-level delegate API to enable XNNPACK, this
method is **NOT RECOMMENDED** unless you need to use TensorFlow Lite both with
and without XNNPACK (e.g. for benchmarking).

With low-level delegate API users create an XNNPACK delegate with the
`TfLiteXNNPackDelegateCreate` function, and then call
`Interpreter::ModifyGraphWithDelegate` to delegate supported parts of
the model to the XNNPACK delegate. The users must destroy the delegate with
`TfLiteXNNPackDelegateDelete` **after** releasing the TensorFlow Lite
interpreter. The snippet below illustrates the typical usage:

```c++
// Build the interpreter
std::unique_ptr<tflite::Interpreter> interpreter;
...

// IMPORTANT: initialize options with TfLiteXNNPackDelegateOptionsDefault() for
// API-compatibility with future extensions of the TfLiteXNNPackDelegateOptions
// structure.
TfLiteXNNPackDelegateOptions xnnpack_options =
    TfLiteXNNPackDelegateOptionsDefault();
xnnpack_options.num_threads = num_threads;

TfLiteDelegate* xnnpack_delegate =
    TfLiteXNNPackDelegateCreate(&xnnpack_options);
if (interpreter->ModifyGraphWithDelegate(xnnpack_delegate) != kTfLiteOk) {
  // Report error and fall back to another delegate, or the default backend
}

...

// Run inference using XNNPACK
interpreter->Invoke()

...

// IMPORTANT: release the interpreter before destroying the delegate
interpreter.reset();
TfLiteXNNPackDelegateDelete(xnnpack_delegate);
```

## Limitations and supported operators

XNNPACK delegate is a work-in-progress, and currently supports a limited set of
operators. Unsupported operators will fall back to the default implementations,
so models using a combination of supported and unsupported operators can still
benefit from XNNPACK delegate.

Below is the list of current operators and limitations:

### `ABS`

* Inputs and outputs must be in 32-bit floating-point format.

### `ADD`

* Inputs and outputs must be in 32-bit floating-point format.
* Only addition with two inputs is supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `AVERAGE_POOL_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* 1x1 pooling is not supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `CEIL`

* Inputs and outputs must be in 32-bit floating-point format.

### `CONV_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `DEPTHWISE_CONV_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `DIV`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `FULLY_CONNECTED`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `FLOOR`

* Inputs and outputs must be in 32-bit floating-point format.

### `HARD_SWISH`

* Inputs and outputs must be in 32-bit floating-point format.

### `LEAKY_RELU`

* Inputs and outputs must be in 32-bit floating-point format.

### `LOGISTIC`

* Inputs and outputs must be in 32-bit floating-point format.

### `MAX_POOL_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* 1x1 pooling is not supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `MAXIMUM`

* Inputs and outputs must be in 32-bit floating-point format.

### `MINIMUM`

* Inputs and outputs must be in 32-bit floating-point format.

### `MUL`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### `NEG`

* Inputs and outputs must be in 32-bit floating-point format.

### `PAD`

* The first input and the output must be in 32-bit floating-point format.
* The second input (the input with the padding specification) must be static
  (use `kTfLiteMmapRo` allocation type).
* The numbers of padding elements must be non-negative.

### `PRELU`

* Inputs and outputs must be in 32-bit floating-point format.
* Slope must be static (use `kTfLiteMmapRo` allocation type).
* Slope must be either a 1D tensor, or have all its non-channel dimensions equal
  1.

### `RELU`

* Inputs and outputs must be in 32-bit floating-point format.

### `RELU6`

* Inputs and outputs must be in 32-bit floating-point format.

### `RELU_N1_TO_1`

* Inputs and outputs must be in 32-bit floating-point format.

### `ROUND`

* Inputs and outputs must be in 32-bit floating-point format.

### `SOFTMAX`

* Inputs and outputs must be in 32-bit floating-point format.
* Only `beta = 1.0` is supported.

### `SQUARE`

* Inputs and outputs must be in 32-bit floating-point format.

### `SQUARED_DIFFERENCE`

* Inputs and outputs must be in 32-bit floating-point format.

### `SUB`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### Other limitations

* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  outputs are not supported.
* Resizing model inputs (via `Interpreter::ResizeInputTensor`) is supported, but
  cause a complete reinitialization of the delegate instance, which has
  considerable overhead.
