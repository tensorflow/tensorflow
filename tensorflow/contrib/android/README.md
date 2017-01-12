# Android TensorFlow support

This directory contains components geared towards supporting TensorFlow on
Android.

## Android Java Inference Interface for TensorFlow

This library contains a Java API capable of loading TensorFlow graphs and
performing inference on Android. See [TensorFlowImageClassifier.java](../../examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java)
in the [TensorFlow Android Camera Demo](../../examples/android) for an example.

With both the native .so library and the Java JAR (or AAR which combines both),
you will have everything you need to use TensorFlow for inference in your
Android app. Note that training is not supported through this interface; for
that you will have to use one of the other APIs.

For prebuilt libraries, see the
[nightly Android build artifacts](https://ci.tensorflow.org/view/Nightly/job/nightly-android/)
page for a recent build.

To build the inference libraries yourself (if, for example, you want to support
custom TensorFlow operators), pick your preferred approach below:

### Bazel

First follow the Bazel setup instructions described in
[tensorflow/examples/android/README.md](../../examples/android/README.md)

Then, to build the native TF library:

```
bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a
```

Replacing `armeabi-v7a` with your desired target architecture.

The library will be located at:

```
bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so
```

To build the Java counterpart:

```
bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
```

You will find the JAR file at:

```
bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar
```

### CMake

For documentation on building a self-contained AAR file with cmake, see
[tensorflow/contrib/android/cmake](cmake).


## AssetManagerFileSystem:

A TensorFlow filesystem supporting the Android asset manager. This may be
useful when writing native (C/C++) code that is tightly coupled with TensorFlow
(for typical usage the Inference Interface library above will be sufficient).
