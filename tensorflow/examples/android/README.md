# TensorFlow Android Camera Demo

This folder contains an example application utilizing TensorFlow for Android
devices.

## Description

The demos in this folder are designed to give straightforward samples of using
TensorFlow in mobile applications. Inference is done using the Java JNI API
exposed by `tensorflow/contrib/android`.

Current samples:

1. [TF Classify](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/ClassifierActivity.java):
        Uses the [Google Inception](https://arxiv.org/abs/1409.4842)
        model to classify camera frames in real-time, displaying the top results
        in an overlay on the camera image.
2. [TF Detect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java):
        Demonstrates a model based on [Scalable Object Detection
        using Deep Neural Networks](https://arxiv.org/abs/1312.2249) to
        localize and track people in the camera preview in real-time.

## To build/install/run

As a prerequisite, Bazel, the Android NDK, and the Android SDK must all be
installed on your system.

1. Get the recommended Bazel version listed at:
        https://www.tensorflow.org/versions/master/get_started/os_setup.html#source
2. The Android NDK may be obtained from:
        http://developer.android.com/tools/sdk/ndk/index.html
3. The Android SDK and build tools may be obtained from:
        https://developer.android.com/tools/revisions/build-tools.html

The Android entries in [`<workspace_root>/WORKSPACE`](../../../WORKSPACE#L2-L13)
must be uncommented with the paths filled in appropriately depending on where
you installed the NDK and SDK. Otherwise an error such as:
"The external label '//external:android/sdk' is not bound to anything" will
be reported.

The TensorFlow `GraphDef` that contains the model definition and weights
is not packaged in the repo because of its size. It will be downloaded
automatically via a new_http_archive defined in WORKSPACE.

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

The associated label and box prior files for the models will also be extracted
into the assets directory.

After editing your WORKSPACE file to update the SDK/NDK configuration,
you may build the APK. Run this from your workspace root:

```bash
$ bazel build //tensorflow/examples/android:tensorflow_demo
```

If you get build errors about protocol buffers, run
`git submodule update --init` and build again.

If adb debugging is enabled on your Android 5.0 or later device, you may then
use the following command from your workspace root to install the APK once
built:

```bash
$ adb install -r -g bazel-bin/tensorflow/examples/android/tensorflow_demo.apk
```

Some older versions of adb might complain about the -g option (returning:
"Error: Unknown option: -g").  In this case, if your device runs Android 6.0 or
later, then make sure you update to the latest adb version before trying the
install command again. If your device runs earlier versions of Android, however,
you can issue the install command without the -g option.

Alternatively, a streamlined means of building, installing and running in one
command is:

```bash
$ bazel mobile-install //tensorflow/examples/android:tensorflow_demo --start_app
```

If camera permission errors are encountered (possible on Android Marshmallow or
above), then the `adb install` command above should be used instead, as it
automatically grants the required camera permissions with `-g`. The permission
errors may not be obvious if the app halts immediately, so if you installed
with bazel and the app doesn't come up, then the easiest thing to do is try
installing with adb.

Once the app is installed it can be started via the "TF Classify" and
"TF Detect" and icons, which have the orange TensorFlow logo as their icon.
