# TF Lite Android App Example

## Building from Source with Bazel

1. Follow the [Bazel steps for the TF Demo App](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#bazel).

2. Build the app with Bazel. The demo needs C++11:

  ```shell
  bazel build -c opt --cxxopt='--std=c++11' \
    //tensorflow/contrib/lite/examples/android:tflite_demo
  ```

3. Install the demo on a
   [debug-enabled device](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install):

  ```shell
  adb install bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk
  ```
