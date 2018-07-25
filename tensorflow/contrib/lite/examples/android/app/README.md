# TF Lite Android App Example

## Building from Source with Bazel

1. Install [Bazel](https://docs.bazel.build/versions/master/install.html), the Android NDK and SDK. The recommended versions are specified on this [webpage](https://www.tensorflow.org/mobile/tflite/demo_android#build_tensorflow_lite_and_the_demo_app_from_source).

2. Build this demo app with Bazel. The demo needs C++11. We configure the fat_apk_cpu flag to package support for 4 hardware variants. You may replace it with --config=android_arm64 on a 64-bit device and --config=android_arm for 32-bit device:

  ```shell
  bazel build -c opt --cxxopt='--std=c++11' --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
    //tensorflow/contrib/lite/examples/android:tflite_demo
  ```

3. Install the demo on a
   [debug-enabled device](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install):

  ```shell
  adb install bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk
  ```
