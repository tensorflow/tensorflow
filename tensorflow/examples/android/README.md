# Tensorflow Android Camera Demo

This folder contains a simple camera-based demo application utilizing Tensorflow.

## Description

This demo uses a Google Inception model to classify camera frames in real-time,
displaying the top results in an overlay on the camera image. See
assets/imagenet_comp_graph_label_strings.txt for the possible classificiations.

## To build/install/run

As a pre-requisite, Bazel, the Android NDK, and the Android SDK must all be
installed on your system. The Android build tools may be obtained from:
https://developer.android.com/tools/revisions/build-tools.html

The Android entries in [<workspace_root>/WORKSPACE](../../WORKSPACE) must be
uncommented with the paths filled in appropriately depending on where you
installed the NDK and SDK. Otherwise an error such as:
"The external label '//external:android/sdk' is not bound to anything" will
be reported.


To build the APK, run this from your workspace root:
```
bazel build //tensorflow/examples/android:tensorflow_demo -c opt --copt=-mfpu=neon
```
Note that "-c opt" is currently required; if not set, an assert (for an
otherwise non-problematic issue) in Eigen will halt the application during
execution. This issue will be corrected in an upcoming release.

If adb debugging is enabled on your device, you may instead use the following
command from your workspace root to automatically build and install:
```
bazel mobile-install //tensorflow/examples/android:tensorflow_demo -c opt --copt=-mfpu=neon
```

Add the "--start_app" flag if you wish to automatically start the app after
installing. Otherwise, find the application icon labeled "Tensorflow Demo".
