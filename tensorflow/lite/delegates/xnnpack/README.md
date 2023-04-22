# XNNPACK backend for TensorFlow Lite

XNNPACK is a highly optimized library of neural network inference operators for
ARM, x86, and WebAssembly architectures in Android, iOS, Windows, Linux, macOS,
and Emscripten environments. This document describes how to use the XNNPACK
library as an inference engine for TensorFlow Lite.

## Using XNNPACK engine with TensorFlow Lite interpreter

XNNPACK integrates with TensorFlow Lite interpreter through the delegation
mechanism. TensorFlow Lite supports several methods to enable XNNPACK
for floating-point inference.

### Enable XNNPACK via Java API on Android (recommended on Android)

Pre-built [nightly TensorFlow Lite binaries for Android](https://www.tensorflow.org/lite/guide/android#use_the_tensorflow_lite_aar_from_mavencentral)
include XNNPACK, albeit it is disabled by default. Use the `setUseXNNPACK`
method in `Interpreter.Options` class to enable it:

```java
Interpreter.Options interpreterOptions = new Interpreter.Options();
interpreterOptions.setUseXNNPACK(true);
Interpreter interpreter = new Interpreter(model, interpreterOptions);
```

### Enable XNNPACK via Swift/Objective-C API on iOS (recommended on iOS)

Pre-built [nightly TensorFlow Lite CocoaPods](https://www.tensorflow.org/lite/guide/ios#specifying_versions)
include XNNPACK, but do not enable it by default. Swift developers can use
`InterpreterOptions` object to enable XNNPACK:

```swift
var options = InterpreterOptions()
options.isXNNPackEnabled = true
var interpreter = try Interpreter(modelPath: "model/path", options: options)
```

Objective-C developers can enable XNNPACK via a new property in the
`TFLInterpreterOptions` class:

```objc
TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
options.useXNNPACK = YES;
NSError *error;
TFLInterpreter *interpreter =
    [[TFLInterpreter alloc] initWithModelPath:@"model/path"
                                      options:options
                                        error:&error];
```

### Enable XNNPACK via Bazel build flags (recommended on desktop)

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

Note that in this case `Interpreter::SetNumThreads` invocation does not take
effect on number of threads used by XNNPACK engine. In order to specify number
of threads available for XNNPACK engine you should manually pass the value when
constructing the interpreter. The snippet below illustrates this assuming you
are using `InterpreterBuilder` to construct the interpreter:

```c++
// Load model
tflite::Model* model;
...

// Construct the interprepter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;

TfLiteStatus res = tflite::InterpreterBuilder(model, resolver, num_threads);
```

**XNNPACK engine used by TensorFlow Lite interpreter uses a single thread for
inference by default.**

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

### Floating-Point Operators

Below is the list of currently supported floating-point operators:

#### `ABS`

* Inputs and outputs must be in 32-bit floating-point format.

#### `ADD`

* Inputs and outputs must be in 32-bit floating-point format.
* Only addition with two inputs is supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `AVERAGE_POOL_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* 1x1 pooling with non-unit stride is not supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `CEIL`

* Inputs and outputs must be in 32-bit floating-point format.

#### `CONV_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `DEPTH_TO_SPACE`

* Inputs and outputs must be in 32-bit floating-point format.
* Block size must be greater than 1.

#### `DEPTHWISE_CONV_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `DIV`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `ELU`

* Inputs and outputs must be in 32-bit floating-point format.

#### `FULLY_CONNECTED`

* Inputs and outputs must be in 32-bit floating-point format.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `FLOOR`

* Inputs and outputs must be in 32-bit floating-point format.

#### `HARD_SWISH`

* Inputs and outputs must be in 32-bit floating-point format.

#### `LEAKY_RELU`

* Inputs and outputs must be in 32-bit floating-point format.

#### `LOGISTIC`

* Inputs and outputs must be in 32-bit floating-point format.

#### `MAX_POOL_2D`

* Inputs and outputs must be in 32-bit floating-point format.
* 1x1 pooling with non-unit stride is not supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `MAXIMUM`

* Inputs and outputs must be in 32-bit floating-point format.

#### `MEAN`

* The first input and the output must be a 4D tensors in 32-bit
  floating-point format.
* The second input (the input with the axes specification) must be static
  (use `kTfLiteMmapRo` allocation type).
* Only [1, 2] or [2, 1] axes specification (i.e. reduction across spatial
  dimensions) is supported.
* Only `keep_dims = True` parameter value is supported.

#### `MINIMUM`

* Inputs and outputs must be in 32-bit floating-point format.

#### `MUL`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `NEG`

* Inputs and outputs must be in 32-bit floating-point format.

#### `PAD`

* The first input and the output must be in 32-bit floating-point format.
* The second input (the input with the padding specification) must be static
  (use `kTfLiteMmapRo` allocation type).
* The numbers of padding elements must be non-negative.

#### `PRELU`

* Inputs and outputs must be in 32-bit floating-point format.
* Slope must be static (use `kTfLiteMmapRo` allocation type).
* Slope must be either a 1D tensor, or have all its non-channel dimensions equal
  1.

#### `RELU`

* Inputs and outputs must be in 32-bit floating-point format.

#### `RELU6`

* Inputs and outputs must be in 32-bit floating-point format.

#### `RELU_N1_TO_1`

* Inputs and outputs must be in 32-bit floating-point format.

#### `RESHAPE`

* The first input and the output must be in 32-bit floating-point format.
* The second input (the input with the new shape specification) must be either
  static (use `kTfLiteMmapRo` allocation type), or absent (with the new shape
  specified via `ReshapeOptions` table).

#### `RESIZE_BILINEAR`

* The first input and the output must be 4D tensors in 32-bit floating-point
  format.
* The second input (the input with the new shape specification) must be
  static (use `kTfLiteMmapRo` allocation type).

#### `ROUND`

* Inputs and outputs must be in 32-bit floating-point format.

#### `SOFTMAX`

* Inputs and outputs must be in 32-bit floating-point format.
* Only `beta = 1.0` is supported.

#### `SQRT`

* Inputs and outputs must be in 32-bit floating-point format.

#### `SQUARE`

* Inputs and outputs must be in 32-bit floating-point format.

#### `SQUARED_DIFFERENCE`

* Inputs and outputs must be in 32-bit floating-point format.

#### `SUB`

* Inputs and outputs must be in 32-bit floating-point format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### Quantized Operators

By default, quantized inference in XNNPACK delegate is disabled, and XNNPACK is
used only for floating-point models. Support for quantized inference in XNNPACK
must be enabled by adding extra Bazel flags when building TensorFlow Lite.

* `--define xnn_enable_qs8=true` flag enables XNNPACK inference for quantized
operators using signed quantization schema. This schema is used by models
produced by [Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
through either post-training integer quantization or quantization-aware
training. Post-training dynamic range quantization is not supported in XNNPACK.

* `--define xnn_enable_qu8=true` flag enables XNNPACK inference for quantized
operators using unsigned quantization schema, produced via the legacy TensorFlow
1.X quantization tooling. This option is experimental and may perform
suboptimally on mobile processors with NEON DOT product instructions.

Below is the list of currently supported quantized operators:

#### `ADD`

* Inputs and outputs must be in 8-bit quantized format.
* Only addition with two inputs is supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `CONV_2D`

* Inputs and outputs must be in 8-bit quantized format (bias must be in 32-bit
  quantized format).
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type),
  and can use either per-tensor or per-channel quantization parameters.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `DEPTHWISE_CONV_2D`

* Inputs and outputs must be in 8-bit quantized format (bias must be in
  32-bit quantized format).
* Bias is mandatory.
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type),
  and can use either per-tensor or per-channel quantization parameters.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `ELU`

* Inputs and outputs must be in 8-bit signed quantized format.

#### `FULLY_CONNECTED`

* Inputs and outputs must be in 8-bit quantized format (bias, if present, must
  be in 32-bit quantized format).
* Both filter and bias must be static (use `kTfLiteMmapRo` allocation type).
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `LOGISTIC`

* Inputs and outputs must be in 8-bit quantized format.

#### `MAX_POOL_2D`

* Inputs and outputs must be in 8-bit quantized format.
* 1x1 pooling with non-unit stride is not supported.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `MUL`

* Inputs and outputs must be in 8-bit quantized format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

#### `PAD`

* The first input and the output must be in 8-bit quantized format.
* The second input (the input with the padding specification) must be static
  (use `kTfLiteMmapRo` allocation type).
* The numbers of padding elements must be non-negative.

#### `SUB`

* Inputs and outputs must be in 8-bit quantized format.
* Fused `NONE`, `RELU`, `RELU_N1_TO_1`, and `RELU6` activations are supported,
  but fused `TANH` and `SIGN_BIT` activations are not.

### Sparse Inference

XNNPACK backend supports sparse inference for CNN models described in the
[Fast Sparse ConvNets](https://arxiv.org/abs/1911.09723) paper. Sparse
inference is restricted to subgraphs with the following operators:

* Sparse subgraph must store its weights in sparse representation (using
  `DENSIFY` operators in the TensorFlow Lite schema).
* Sparse subgraph must start with a 3x3 stride-2 `CONV_2D` operator with
  padding 1 on each side, no dilation, and 3 input channels.
* Sparse subgraph must end with either a `MEAN` operator with reduction across
  spatial axes, or a `DEPTH_TO_SPACE` operator.
* Sparse subgraph may contain the following operators:
  * `CONV_2D` with 1x1 kernel and no padding. At least 2/3rd of filter weights
    in the 1x1 `CONV_2D` operators across the sparse subgraph must be zeroes
    to enable sparse inference.
  * `DEPTHWISE_CONV_2D` with 3x3 kernel, stride 1, no dilation, and padding 1
    on each side.
  * `DEPTHWISE_CONV_2D` with 3x3 kernel, stride 2, no dilation, and padding 1
    on each side.
  * `DEPTHWISE_CONV_2D` with 5x5 kernel, stride 1, no dilation, and padding 2
    on each side.
  * `DEPTHWISE_CONV_2D` with 5x5 kernel, stride 2, no dilation, and padding 2
    on each side.
  * `RESIZE_BILINEAR` operator with output dimensions greater than 1.
  * `MEAN` operator with reduction across spatial axes.
  * `ADD` and `MUL` operators where both inputs are 4D tensors. If one of the
    inputs to `ADD` or `MUL` is a constant tensor, it must be representable as
    either a scalar, or a 1D vector.
  * Unary elementwise operators `ABS`, `CEIL`, `ELU`, `FLOOR`, `HARD_SWISH`,
    `LEAKY_RELU`, `LOGISTIC`, `NEG`, `RELU`, `RELU6`, `RELU_N1_TO_1`, `ROUND`,
    `SIGMOID`, and `SQUARE`.

Pre-trained [Fast Sparse ConvNets models](https://github.com/google-research/google-research/tree/master/fastconvnets)
provide examples that satisfy these constrains.

### Other limitations

* Dynamically allocated (with `kTfLiteDynamic` allocation type) inputs and
  outputs are not supported.
* Resizing model inputs (via `Interpreter::ResizeInputTensor`) is supported, but
  cause a complete reinitialization of the delegate instance, which has
  considerable overhead.
