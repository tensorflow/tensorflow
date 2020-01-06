## Tensorflow Lite Hexagon Delegate Quick Guide

[TOC]

This document explains how to use the Tensorflow Lite Hexagon Delegate in your
application using the Java and/or C API. The delegate leverages the Qualcomm
Hexagon library to execute quantized kernels on the DSP. Note that the delegate
is intended to *complement* NNAPI functionality, particularly for devices where
NNAPI DSP acceleration is unavailable (e.g., on older devices, or devices that
don’t yet have a DSP NNAPI driver). Note: This delegate is in experimental
(beta) phase.

**Supported devices:**

Currently most
[Qualcomm SoCs](https://en.wikipedia.org/wiki/List_of_Qualcomm_Snapdragon_systems-on-chip)
are supported, including:

*   Snapdragon 835 (682 DSP)
*   Snapdragon 660/820/821 (680 DSP)
*   Snapdragon 710/845 (685 DSP)
*   Snapdragon 8150/855 (690 DSP)

**Supported models:**

The Hexagon delegate currently supports quantized models generated using
[quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize),
e.g.,
[these quantized models](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models)
hosted on the TensorFlow Lite repo. It does not (yet) support models with
[8-bit symmetric quantization spec](https://www.tensorflow.org/lite/performance/quantization_spec).
Sample models include
[MobileNet V1](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz),
[SSD Mobilenet](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip).

## Hexagon Delegate Java API

```
public class HexagonDelegate implements Delegate, Closeable {

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not
   * available on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException


  /**
   * Frees TFLite resources in C runtime.
   *
   * User is expected to call this method explicitly.
   */
  @Override
  public void close();
}
```

## Example Usage from Java

NOTE: As of 19 Dec 2019 you need to use the nightly build for TFLite (typically
imported in gradle via `implementation
'org.tensorflow:tensorflow-lite:0.0.0-nightly'`).

1.  Add the ‘tensorflow-lite-hexagon.aar’ to your app - this is in addition to
    the standard tensorflow-lite AAR (nightly or release).
    [Relevant instructions](https://stackoverflow.com/questions/16682847/how-to-manually-include-external-aar-package-using-new-gradle-android-build-syst).
    You can do this by running bazel command like example below for arm64. We
    will provide a version hosted on JCenter soon.
    *   `bazel build -c opt --config=android_arm64
        tensorflow/lite/experimental/delegates/hexagon/java:tensorflow-lite-hexagon`
1.  Download and run
    [“hexagon_nn_skel.run](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)” -
    Note: you will need to accept the license agreement. It should provide 3
    different shared libraries “libhexagon_nn_skel.so”,
    “libhexagon_nn_skel_v65.so”, “libhexagon_nn_skel_v66.so” \
    Include all 3 in your app with other shared libraries. See
    [How to add shared library to your app](#how-to-add-shared-library-to-your-app)
    \
    The delegate will automatically pick the one with best performance depending
    on the device. \
    Note: If your app will be built for both 32 and 64-bit ARM devices, then you
    will need to add the hexagon shared libs to both 32 and 64-bit lib folders.

1.  Create a delegate, example:

```
import org.tensorflow.lite.experimental.HexagonDelegate;

// Create the Delegate instance.
try {
  hexagonDelegate = new HexagonDelegate(activity);
  tfliteOptions.addDelegate(hexagonDelegate);
} catch (UnsupportedOperationException e) {
  // Hexagon delegate is not supported on this device.
}

tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);

// Dispose after finished with inference.
tfliteInterpreter.close();
if (hexagonDelegate != null) {
  hexagonDelegate.close();
}
```

## Hexagon Delegate C API

```
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;
  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;
  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate*
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
Void TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
Void TfLiteHexagonTearDown();
```

## Example Usage from C

1.  Add the ‘tensorflow-lite-hexagon.aar’ to your app - this is in addition to
    the standard tensorflow-lite AAR (nightly or release).
    [Relevant instructions](https://stackoverflow.com/questions/16682847/how-to-manually-include-external-aar-package-using-new-gradle-android-build-syst).
1.  Include the provided hexagon_delegate.h
1.  Download and run
    [“hexagon_nn_skel.run](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)” -
    Note: you will need to accept the license agreement. It should provide 3
    different shared libraries \
    “libhexagon_nn_skel.so”, “libhexagon_nn_skel_v65.so”,
    “libhexagon_nn_skel_v66.so” \
    Include all 3 in your app with other shared libraries. See How to add shared
    library to your app. \
    The delegate will automatically pick the one with best performance depending
    on the device. \
    Note: If your app will be built for both 32 and 64-bit ARM devices, then you
    will need to add the hexagon shared libs to both 32 and 64-bit lib folders.

1.  In your code, ensure the native Hexagon library is loaded. This can be done
    by calling `System.loadLibrary("tensorflowlite_hexagon_jni");` \
    in your Activity or Java entry-point.

1.  Create a delegate, example:

    ```
    #include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h"

    // Assuming shared libraries are under "/data/local/tmp/"
    // If files are packaged with native lib in android App then it
    // will typically be equivalent to the path provided by
    // "getContext().getApplicationInfo().nativeLibraryDir"
    const char[] library_directory_path = "/data/local/tmp/";
    TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
    ::tflite::TfLiteHexagonDelegateOptions params = {0};
    // 'delegate_ptr' Need to outlive the interpreter. For example,
    // If use case will need to resize input or anything that can trigger
    // re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
    auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
    Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
      [](TfLiteDelegate* delegate) {
        ::tflite::TfLiteHexagonDelegateDelete(delegate);
      });
    interpreter->ModifyGraphWithDelegate(delegate.get());
    // After usage of delegate.
    TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
    ```

## How to add shared library to your app

Create folder “app/src/main/jniLibs”, then for each target architecture create a
directory.

For example,

Arm64 bit: “app/src/main/jniLibs/arm64-v8a”

Arm32 bit: “app/src/main/jniLibs/armeabi-v7a”

Put your .so in the directory that match the architecture.

## Feedback

For issues, please create a
[github](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md)
issue with all the necessary repro details, including the phone model and board
used (`adb shell getprop ro.product.device` and `adb shell getprop
ro.board.platform`).

## FAQ

*   Will the delegate support models created using
    [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)?
    *   This is tentatively planned for a future release, though there is no
        concrete timeline.
*   Which ops are supported by the delegate?
    *   Initial Dogfood list of supported ops:
        *   Add
        *   ArgMax
        *   ArgMin
        *   AveragePool2D (without any activation)
        *   Concat
        *   Conv2D w/ following constraints:
          *   stride width/height <= 3
        *   DepthwiseConv2D w/ following constraints:
          *   Filter width == 3
          *   depth_multiplier == 1
          *   dilation only supported when stride == 1
          *   Otherwise, stride height/width <= 3
        *   FullyConnected (without any activation)
        *   L2Normalization (without any activation)
        *   Logistic (aka Sigmoid)
        *   MaxPool2D (without any activation)
        *   Mul (without any activation)
        *   Neg
        *   Pad: Only supports 0 padding
        *   Relu
        *   Relu6
        *   Reshape
        *   Resize Bilinear w/ following constraints:
          *   Requested size <= 65
        *   Resize Nearest Neighbor
        *   SoftMax
        *   Split
        *   Sub
        *   Tanh
        *   Transpose
        *   TransposeConv2D w/ following constraints:
          *   stride height/width <= 3
          *   dilation height/width == 1
*   How can I tell that the model is using the DSP when I enable the delegate?
    *   A log message will be printed whether delegate created or not, and
        another one with how many nodes are running using the delegate. \
        "Created TensorFlow Lite delegate for Hexagon." \
        "Hexagon delegate: X nodes delegated out of Y nodes."
*   Do I need all Ops in the model to be supported to run the delegate ?
    *   No, the Model will be partitioned into subgraphs based on the supported
        ops. Any unsupported ops will run on the CPU.
