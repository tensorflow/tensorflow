# TF Lite Android Image Classifier App Example

A simple Android example that demonstrates image classification using the camera.

## Building in Android Studio with TensorFlow Lite AAR from JCenter.
The build.gradle is configured to use TensorFlow Lite's nightly build.

If you see a build error related to compatibility with Tensorflow Lite's Java API (example: method X is
undefined for type Interpreter), there has likely been a backwards compatible
change to the API. You will need to pull new app code that's compatible with the
nightly build and may need to first wait a few days for our external and internal
code to merge.

## Building from Source with Bazel

1. Follow the [Bazel steps for the TF Demo App](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#bazel):

  1. [Install Bazel and Android Prerequisites](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install-bazel-and-android-prerequisites).
     It's easiest with Android Studio.

      - You'll need at least SDK version 23.
      - Make sure to install the latest version of Bazel. Some distributions
        ship with Bazel 0.5.4, which is too old.
      - Bazel requires Android Build Tools `28.0.0` or higher.
      - You also need to install the Android Support Repository, available
        through Android Studio under `Android SDK Manager -> SDK Tools ->
        Android Support Repository`.

  2. [Edit your `WORKSPACE`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#edit-workspace)
     to add SDK and NDK targets.

     NOTE: As long as you have the SDK and NDK installed, the `./configure`
     script will create these rules for you. Answer "Yes" when the script asks
     to automatically configure the `./WORKSPACE`.

      - Make sure the `api_level` in `WORKSPACE` is set to an SDK version that
        you have installed.
      - By default, Android Studio will install the SDK to `~/Android/Sdk` and
        the NDK to `~/Android/Sdk/ndk-bundle`.

2. Build the app with Bazel. The demo needs C++11:

  ```shell
  bazel build -c opt --cxxopt='--std=c++11' \
    //tensorflow/lite/java/demo/app/src/main:TfLiteCameraDemo
  ```

3. Install the demo on a
   [debug-enabled device](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install):

  ```shell
  adb install bazel-bin/tensorflow/lite/java/demo/app/src/main/TfLiteCameraDemo.apk
  ```
