# Development tools for Android

TensorFlow Lite provides a number of tools for integrating models into Android
apps. This page describes development tools for use in building apps with
Kotlin, Java, and C++, as well as support for TensorFlow Lite development in
Android Studio.

Key Point: In general, you should use the [TensorFlow Lite Task
Library](#task_library) for integrating TensorFlow Lite into your Android app,
unless your use case is not supported by that library. If it's not supported by
the Task Library, use the [TensorFlow Lite library](#lite_lib) and [Support
library](#support_lib).

To get started quickly writing Android code, see the
[Quickstart for Android](../android/quickstart)


## Tools for building with Kotlin and Java

The following sections describe development tools for TensorFlow Lite that use
the Kotlin and Java languages.

### TensorFlow Lite Task Library {:#task_library}

TensorFlow Lite Task Library contains a set of powerful and easy-to-use
task-specific libraries for app developers to build with TensorFlow Lite.
It provides optimized out-of-box model interfaces for popular machine learning
tasks, such as image classification, question and answer, etc. The model
interfaces are specifically designed for each task to achieve the best
performance and usability. Task Library works cross-platform and is supported on
Java and C++.

To use the Task Library in your Android app, use the AAR from MavenCentral for
[Task Vision library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision)
,
[Task Text library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text)
and
[Task Audio Library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-audio)
, respectively.

You can specify this in your `build.gradle` dependencies as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.3.0'
}
```

If you use nightly snapshots, make sure you add the
[Sonatype snapshot repository](./lite_build#use_nightly_snapshots) to your
project.

See the introduction in the
[TensorFlow Lite Task Library overview](../inference_with_metadata/task_library/overview.md)
for more details.

### TensorFlow Lite library {:#lite_lib}

Use the TensorFlow Lite library in your Android app by adding the
[AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite)
to your development project.

You can specify this in your `build.gradle` dependencies as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
}
```

If you use nightly snapshots, make sure you add the
[Sonatype snapshot repository](./lite_build#use_nightly_snapshots) to your
project.

This AAR includes binaries for all of the
[Android ABIs](https://developer.android.com/ndk/guides/abis). You can reduce
the size of your application's binary by only including the ABIs you need to
support.

Unless you are targeting specific hardware, you should omit the `x86`, `x86_64`,
and `arm32` ABIs in most cases. You can configure this with the following Gradle
configuration. It specifically includes only `armeabi-v7a` and `arm64-v8a`, and
should cover most modern Android devices.

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
[Android ABIs](https://developer.android.com/ndk/guides/abis)
in the Android NDK documentation.

### TensorFlow Lite Support Library {:#support_lib}

The TensorFlow Lite Android Support Library makes it easier to integrate models
into your application. It provides high-level APIs that help transform raw input
data into the form required by the model, and interpret the model's output,
reducing the amount of boilerplate code required.

It supports common data formats for inputs and outputs, including images and
arrays. It also provides pre- and post-processing units that perform tasks such
as image resizing and cropping.

Use the Support Library in your Android app by including the TensorFlow Lite
[Support Library AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support).

You can specify this in your `build.gradle` dependencies as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.3.0'
}
```

If you use nightly snapshots, make sure you add the
[Sonatype snapshot repository](./lite_build#use_nightly_snapshots) to your
project.

For instructions on how to get started, see the
[TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md).

### Minimum Android SDK versions for libraries

| Library                     | `minSdkVersion` | Device Requirements    |
| --------------------------- | --------------- | ---------------------- |
| tensorflow-lite             | 19              | NNAPI usage requires   |
:                             :                 : API 27+                :
| tensorflow-lite-gpu         | 19              | GLES 3.1 or OpenCL     |
:                             :                 : (typically only        :
:                             :                 : available on API 21+   :
| tensorflow-lite-hexagon     | 19              | -                      |
| tensorflow-lite-support     | 19              | -                      |
| tensorflow-lite-task-vision | 21              | android.graphics.Color |
:                             :                 : related API requires   :
:                             :                 : API 26+                :
| tensorflow-lite-task-text   | 21              | -                      |
| tensorflow-lite-task-audio  | 23              | -                      |
| tensorflow-lite-metadata    | 19              | -                      |

### Using Android Studio

In addition to the development libraries described above, Android Studio
also provides support for integrating TensorFlow Lite models, as described
below.

#### Android Studio ML Model Binding

The ML Model Binding feature of Android Studio 4.1 and later allows you to
import `.tflite` model files into your existing Android app, and generate
interface classes to make it easier to integrate your code with a model.

To import a TensorFlow Lite (TFLite) model:

1.  Right-click on the module you would like to use the TFLite model or click on
    **File > New > Other > TensorFlow Lite Model**.

1.  Select the location of your TensorFlow Lite file. Note that the tooling
    configures the module's dependency with ML Model binding and
    automatically adds all required dependencies to your Android module's
    `build.gradle` file.

    Note: Select the second checkbox for importing TensorFlow GPU if you
    want to use [GPU acceleration](../performance/gpu).

1.  Click `Finish` to begin the import process. When the import is finished, the
    tool displays a screen describing the model, including its input and output
    tensors.

1.  To start using the model, select Kotlin or Java, copy and paste the code
    in the **Sample Code** section.

You can return to the model information screen by double clicking the TensorFlow
Lite model under the `ml` directory in Android Studio. For more information on
using the Modle Binding feature of Android Studio, see the Android Studio
[release notes](https://developer.android.com/studio/releases#4.1-tensor-flow-lite-models).
For an overview of using model binding in Android Studio, see the code example
[instructions](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md).


## Tools for building with C and C++

The C and C++ libraries for TensorFlow Lite are primarily intended for
developers using the Android Native Development Kit (NDK) to build their apps.
There are two ways to use TFLite through C++ if you build your app with the NDK:

### TFLite C API

Using this API is the *recommended* approach for developers using the NDK.
Download the
[TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite)
file, rename to `tensorflow-lite-*.zip`, and unzip it. You must include the four
header files in the `headers/tensorflow/lite/` and `headers/tensorflow/lite/c/`
folders and the relevant `libtensorflowlite_jni.so` dynamic library in the `jni/`
folder in your NDK project.

The `c_api.h` header file contains basic documentation about using the TFLite C
API.

### TFLite C++ API

If you want to use TFLite through C++ API, you can build the C++ shared
libraries:

32bit armeabi-v7a:

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64bit arm64-v8a:

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

Currently, there is no straightforward way to extract all header files needed,
so you must include all header files in `tensorflow/lite/` from the TensorFlow
repository. Additionally, you will need header files from
[FlatBuffers](https://github.com/google/flatbuffers) and
[Abseil](https://github.com/abseil/abseil-cpp).
