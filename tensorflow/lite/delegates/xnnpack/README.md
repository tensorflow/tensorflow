# XNNPACK backend for TensorFlow Lite

XNNPACK is a highly optimized library of floating-point neural network
inference operators for ARM, WebAssembly, and x86 platforms. This document
describes how to use the XNNPACK library as a backend for TensorFlow Lite.

## Enabling XNNPACK backend in TensorFlow Lite models

XNNPACK integrates with TensorFlow Lite interpreter through the delegation
mechanism. To leverage XNNPACK library for acceleration, the users need to
create an XNNPACK delegate with the `TfLiteXNNPackDelegateCreate` function,
and call `Interpreter::ModifyGraphWithDelegate` to delegate supported parts of
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

// IMPORTANT: release the interpreter before destroing the delegate
interpreter.reset();
TfLiteXNNPackDelegateDelete(xnnpack_delegate);
```

## Limitations and supported operators

XNNPACK delegate is a work-in-progress, and currently supports a limited set of
operators. Unsupported operators will fall back to the default implementations,
so models using a combination of supported and unsupported operators can still
benefit from XNNPACK delegate.

Below is the list of current operators and limitations:

### `ADD`

* Inputs and outputs must be in 32-bit floating-point format.
* Only addition with two inputs is supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `CONV_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) input and output
  are not supported.

### `DEPTHWISE_CONV_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) input and output
  are not supported.

### `HARD_SWISH`

* Inputs and outputs must be in 32-bit floating-point format.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `LOGISTIC`

* Inputs and outputs must be in 32-bit floating-point format.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `MUL`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `PRELU`

* Inputs and outputs must be in 32-bit floating-point format.
* Slope must be static (use `kTfLiteMmapRo` allocation type).
* Slope must be either a 1D tensor, or have all its non-channel dimensions equal
  1.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) input and output
  are not supported.

### `RELU`

* Inputs and outputs must be in 32-bit floating-point format.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `RELU6`

* Inputs and outputs must be in 32-bit floating-point format.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `RELU_N1_TO_1`

* Inputs and outputs must be in 32-bit floating-point format.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### `SOFTMAX`

* Inputs and outputs must be in 32-bit floating-point format.
* Only `beta = 1.0` is supported.
* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  output are not supported.

### Other limitations

* Resizing model inputs (via `Interpreter::ResizeInputTensor`) is supported, but
  cause a complete reinitialization of the delegate instance, which has
  considerable overhead.
