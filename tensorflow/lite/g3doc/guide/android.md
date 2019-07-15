# Android quickstart

To get started with TensorFlow Lite on Android, we recommend exploring the
following example.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android
image classification example</a>

For an explanation of the source code, you should also read
[TensorFlow Lite Android image classification](https://www.tensorflow.org/lite/models/image_classification/android).

This example app uses
[image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
to continuously classify whatever it sees from the device's rear-facing camera.
The application can run either on device or emulator.

Inference is performed using the TensorFlow Lite Java API. The demo app
classifies frames in real-time, displaying the top most probable
classifications. It allows the user to choose between a floating point or
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
model, select the thread count, and decide whether to run on CPU, GPU, or via
[NNAPI](https://developer.android.com/ndk/guides/neuralnetworks).

Note: Additional Android applications demonstrating TensorFlow Lite in a variety
of use cases are available in
[Examples](https://www.tensorflow.org/lite/examples).

## Build in Android Studio

To build the example in Android Studio, follow the instructions in
[README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md).

## Create your own Android app

To get started quickly writing your own Android code, we recommend using our
[Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)
as a starting point.

The following sections contain some useful information for working with
TensorFlow Lite on Android.

### Use the TensorFlow Lite AAR from JCenter

To use TensorFlow Lite in your Android app, we recommend using the
[TensorFlow Lite AAR hosted at JCenter](https://bintray.com/google/tensorflow/tensorflow-lite).

You can specify this in your `build.gradle` dependencies as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
}
```

This AAR includes binaries for all of the
[Android ABIs](https://developer.android.com/ndk/guides/abis). You can reduce
the size of your application's binary by only including the ABIs you need to
support.

We recommend most developers omit the `x86`, `x86_64`, and `arm32` ABIs. This
can be achieved with the following Gradle configuration, which specifically
includes only `armeabi-v7a` and `arm64-v8a`, which should cover most modern
Android devices.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

To learn more about `abiFilters`, see
[`NdkOptions`](https://google.github.io/android-gradle-dsl/current/com.android.build.gradle.internal.dsl.NdkOptions.html)
in the Android Gradle documentation.

### Build TensorFlow Lite locally

In some cases, you might wish to use a local build of TensorFlow Lite. For
example, you may be building a custom binary that includes
[operations selected from TensorFlow](https://www.tensorflow.org/lite/guide/ops_select).

In this case, follow the
[custom AAR build instructions](https://www.tensorflow.org/lite/guide/ops_select#android_aar)
to create your own AAR and include it in your app.
