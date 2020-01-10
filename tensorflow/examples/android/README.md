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

If adb debugging is enabled on your Android 5.0 or later device, you may then
use the following command from your workspace root to install the APK once
built:
'''
adb install -r -g bazel-bin/tensorflow/examples/android/tensorflow_demo_incremental.apk
'''

Alternatively, a streamlined means of building, installing and running in one
command is:
```
bazel mobile-install //tensorflow/examples/android:tensorflow_demo -c opt --start_app --copt=-mfpu=neon
```

If camera permission errors are encountered (possible on Android Marshmallow or
above), then the adb install command above should be used instead, as it
automatically grants the required camera permissions with '-g'.
