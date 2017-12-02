# Android TensorFlow support

This directory defines components (a native `.so` library and a Java JAR)
geared towards supporting TensorFlow on Android. This includes:

- The [TensorFlow Java API](../../java/README.md)
- A `TensorFlowInferenceInterface` class that provides a smaller API
  surface suitable for inference and summarizing performance of model execution.

For example usage, see [TensorFlowImageClassifier.java](../../examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java)
in the [TensorFlow Android Demo](../../examples/android).

For prebuilt libraries, see the
[nightly Android build artifacts](https://ci.tensorflow.org/view/Nightly/job/nightly-android/)
page for a recent build.

The TensorFlow Inference Interface is also available as a
[JCenter package](https://bintray.com/google/tensorflow/tensorflow)
(see the tensorflow-android directory) and can be included quite simply in your
android project with a couple of lines in the project's `build.gradle` file:

```
allprojects {
    repositories {
        jcenter()
    }
}

dependencies {
    compile 'org.tensorflow:tensorflow-android:+'
}
```

This will tell Gradle to use the
[latest version](https://bintray.com/google/tensorflow/tensorflow-android/_latestVersion)
of the TensorFlow AAR that has been released to
[https://bintray.com/google/tensorflow/tensorflow-android](https://bintray.com/google/tensorflow/tensorflow-android).
You may replace the `+` with an explicit version label if you wish to
use a specific release of TensorFlow in your app.

To build the libraries yourself (if, for example, you want to support custom
TensorFlow operators), pick your preferred approach below:

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


## AssetManagerFileSystem

This directory also contains a TensorFlow filesystem supporting the Android
asset manager. This may be useful when writing native (C++) code that is tightly
coupled with TensorFlow. For typical usage, the library above will be
sufficient.
