# TensorFlow Lite Hexagon delegate

This document explains how to use the TensorFlow Lite Hexagon Delegate in your
application using the Java and/or C API. The delegate leverages the Qualcomm
Hexagon library to execute quantized kernels on the DSP. Note that the delegate
is intended to *complement* NNAPI functionality, particularly for devices where
NNAPI DSP acceleration is unavailable (e.g., on older devices, or devices that
don’t yet have a DSP NNAPI driver).

Note: This delegate is in experimental (beta) phase.

**Supported devices:**

Currently most
[Qualcomm SoCs](https://en.wikipedia.org/wiki/List_of_Qualcomm_Snapdragon_systems-on-chip)
are supported, including:

*   Snapdragon 835 (682 DSP)
*   Snapdragon 660/820/821 (680 DSP)
*   Snapdragon 710/845 (685 DSP)
*   Snapdragon 8150/855 (690 DSP)

**Supported models:**

The Hexagon delegate supports all models that conform to our
[8-bit symmetric quantization spec](https://www.tensorflow.org/lite/performance/quantization_spec),
including those generated using
[post-training integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant).
UInt8 models trained with the legacy
[quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)
path are also supported, for e.g.,
[these quantized versions](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models)
on our Hosted Models page.

## Hexagon delegate Java API

```java
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

### Example usage

#### Step 1. Edit app/build.gradle to use the nightly Hexagon delegate AAR

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly'
}
```

#### Step 2. Add Hexagon libraries to your Android app

*   Download and run hexagon_nn_skel.run. It should provide 3 different shared
    libraries “libhexagon_nn_skel.so”, “libhexagon_nn_skel_v65.so”,
    “libhexagon_nn_skel_v66.so”
    *   [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    *   [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    *   [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)

Note: You will need to accept the license agreement.

Note: As of 04/28/2020 you should use v1.17.

Note: You must use the hexagon_nn libraries with the compatible version of
interface library. Interface library is part of the AAR and fetched by bazel
through the
[config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl)
The version in the bazel config is the version you should use.

*   Include all 3 in your app with other shared libraries. See
    [How to add shared library to your app](#how-to-add-shared-library-to-your-app).
    The delegate will automatically pick the one with best performance depending
    on the device.

Note: If your app will be built for both 32 and 64-bit ARM devices, then you
will need to add the Hexagon shared libs to both 32 and 64-bit lib folders.

#### Step 3. Create a delegate and initialize a TensorFlow Lite Interpreter

```java
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

## Hexagon delegate C API

```c
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the Hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the Hexagon SDK.
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

### Example usage

#### Step 1. Edit app/build.gradle to use the nightly Hexagon delegate AAR

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly'
}
```

#### Step 2. Add Hexagon libraries to your Android app

*   Download and run hexagon_nn_skel.run. It should provide 3 different shared
    libraries “libhexagon_nn_skel.so”, “libhexagon_nn_skel_v65.so”,
    “libhexagon_nn_skel_v66.so”
    *   [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    *   [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)

Note: You will need to accept the license agreement.

Note: As of 03/03/2020 you should use v1.14.

Note: You must use the hexagon_nn libraries with the compatible version of
interface library. Interface library is part of the AAR and fetched by bazel
through the
[config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl).
The version in the bazel config is the version you should use.

*   Include all 3 in your app with other shared libraries. See
    [How to add shared library to your app](#how-to-add-shared-library-to-your-app).
    The delegate will automatically pick the one with best performance depending
    on the device.

Note: If your app will be built for both 32 and 64-bit ARM devices, then you
will need to add the Hexagon shared libs to both 32 and 64-bit lib folders.

#### Step 3. Include the C header

*   The header file "hexagon_delegate.h" can be downloaded from
    [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h)
    or extracted from the Hexagon delegate AAR.

#### Step 4. Create a delegate and initialize a TensorFlow Lite Interpreter

*   In your code, ensure the native Hexagon library is loaded. This can be done
    by calling `System.loadLibrary("tensorflowlite_hexagon_jni");` \
    in your Activity or Java entry-point.

*   Create a delegate, example:

```c
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

## Add the shared library to your app

*   Create folder “app/src/main/jniLibs”, and create a directory for each target
    architecture. For example,
    *   ARM 64-bit: `app/src/main/jniLibs/arm64-v8a`
    *   ARM 32-bit: `app/src/main/jniLibs/armeabi-v7a`
*   Put your .so in the directory that match the architecture.

Note: If you're using App Bundle for publishing your Application, you might want
to set android.bundle.enableUncompressedNativeLibs=false in the
gradle.properties file.

## Feedback

For issues, please create a
[GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md)
issue with all the necessary repro details, including the phone model and board
used (`adb shell getprop ro.product.device` and `adb shell getprop
ro.board.platform`).

## FAQ

*   Which ops are supported by the delegate?
    *   See the current list of
        [supported ops and constraints](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/hexagon/README.md)
*   How can I tell that the model is using the DSP when I enable the delegate?
    *   Two log messages will be printed when you enable the delegate - one to
        indicate if the delegate was created and another to indicate how many
        nodes are running using the delegate. \
        `Created TensorFlow Lite delegate for Hexagon.` \
        `Hexagon delegate: X nodes delegated out of Y nodes.`
*   Do I need all Ops in the model to be supported to run the delegate?
    *   No, the Model will be partitioned into subgraphs based on the supported
        ops. Any unsupported ops will run on the CPU.
*   How can I build the Hexagon delegate AAR from source?
    *   Use `bazel build -c opt --config=android_arm64
        tensorflow/lite/experimental/delegates/hexagon/java:tensorflow-lite-hexagon`.
*   Why does Hexagon delegate fail to initialize although my Android device has
    a supported SoC?
    *   Verify if your device indeed has a supported SoC. Run `adb shell cat
        /proc/cpuinfo | grep Hardware` and see if it returns something like
        "Hardware : Qualcomm Technologies, Inc MSMXXXX".
    *   Some phone manufacturers use different SoCs for the same phone model.
        Therefore, Hexagon delegate may only work on some but not all devices of
        the same phone model.
    *   Some phone manufactures intentionally restrict the use of Hexagon DSP
        from non-system Android apps, making the Hexagon delegate unable to
        work.
