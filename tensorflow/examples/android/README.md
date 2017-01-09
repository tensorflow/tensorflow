# TensorFlow Android Camera Demo

This folder contains an example application utilizing TensorFlow for Android
devices.

## Description

The demos in this folder are designed to give straightforward samples of using
TensorFlow in mobile applications.

Inference is done using the [TensorFlow Android Inference Interface](../../../tensorflow/contrib/android),
which may be built separately if you want a standalone library to drop into your
existing application.

# Current samples:

1. [TF Classify](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/ClassifierActivity.java):
        Uses the [Google Inception](https://arxiv.org/abs/1409.4842)
        model to classify camera frames in real-time, displaying the top results
        in an overlay on the camera image.
2. [TF Detect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java):
        Demonstrates a model based on [Scalable Object Detection
        using Deep Neural Networks](https://arxiv.org/abs/1312.2249) to
        localize and track people in the camera preview in real-time.

## Prebuilt APK:

If you just want the fastest path to trying the demo, you may download the
nightly build [here](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-android/TF_BUILD_CONTAINER_TYPE=ANDROID,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=NO_PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=android-slave/).
A device running Android 5.0 (API 21) or higher is required.

## Running the Demo

Once the app is installed it can be started via the "TF Classify" and
"TF Detect" and icons, which have the orange TensorFlow logo as their icon.

While running the activities, pressing the volume keys on your device will
toggle debug visualizations on/off, rendering additional info to the screen
that may be useful for development purposes.

## Building the Demo from Source

### Install Bazel and Android Prerquisites

Bazel is the primary build system for TensorFlow. To build with Bazel,
it and the Android NDK and SDK must be installed on your system.

1. Get the recommended Bazel version listed at:
        https://www.tensorflow.org/versions/master/get_started/os_setup.html#source
2. The Android NDK is required to build the native (C/C++) TensorFlow code.
        The current recommended version is 12b, which may be found
        [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-12b-downloads).
3. The Android SDK and build tools may be obtained from:
        https://developer.android.com/tools/revisions/build-tools.html
        The SDK also comes packaged with Android Studio, if you have that
        installed.

#### Edit WORKSPACE

The Android entries in [`<workspace_root>/WORKSPACE`](../../../WORKSPACE#L2-L13)
must be uncommented with the paths filled in appropriately depending on where
you installed the NDK and SDK. Otherwise an error such as:
"The external label '//external:android/sdk' is not bound to anything" will
be reported.

Also edit the API levels for the SDK in WORKSPACE to the highest level you
have installed in your SDK. This must be >= 23 (this is completely independent
of the API level of the demo, which is defined in AndroidManifest.xml).
The NDK API level may remain at 21.

#### Install Model Files (optional)

The TensorFlow `GraphDef`s that contain the model definitions and weights
are not packaged in the repo because of their size. They are downloaded
automatically and packaged with the APK by Bazel via a new_http_archive defined
in WORKSPACE during the build process.

**Optional**: If you wish to place the models in your assets manually (E.g. for
non-Bazel builds), remove the `inception_5` and `mobile_multibox` entries in
`BUILD` and download the archives yourself to the `assets` directory in the
source tree:

```bash
$ curl -L https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -o /tmp/inception5h.zip
$ curl -L https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1.zip -o /tmp/mobile_multibox_v1.zip

$ unzip /tmp/inception5h.zip -d tensorflow/examples/android/assets/
$ unzip /tmp/mobile_multibox_v1.zip -d tensorflow/examples/android/assets/
```

This will extract the models and their associated metadata files to the local
assets/ directory.

After editing your WORKSPACE file to update the SDK/NDK configuration,
you may build the APK. Run this from your workspace root:

#### Build

```bash
$ bazel build -c opt //tensorflow/examples/android:tensorflow_demo
```

If you get build errors about protocol buffers, run
`git submodule update --init` and make sure that you've modified your WORKSPACE
file as instructed, then try building again.

#### Install

Make sure that adb debugging is enabled on your Android 5.0 (API 21) or
later device, then after building use the following command from your workspace
root to install the APK:

```bash
$ adb install -r bazel-bin/tensorflow/examples/android/tensorflow_demo.apk
```

### Android Studio

Android Studio may be used to build the demo in conjunction with Bazel. First,
make sure that you can build with Bazel following the above directions. Then,
look at (build.gradle)[build.gradle] and make sure that the path to Bazel
matches that of your system.

At this point you can add the tensorflow/examples/android directory as a new
Android Studio project. Click through installing all the Gradle extensions it
requests, and you should be able to have Android Studio build the demo like any
other application (it will call out to Bazel to build the native code with the
NDK).

### CMake

Full CMake support for the demo is coming soon, but for now it is possible to
build the TensorFlow Android Inference library using
[tensorflow/contrib/android/cmake](../../../tensorflow/contrib/android/cmake).
