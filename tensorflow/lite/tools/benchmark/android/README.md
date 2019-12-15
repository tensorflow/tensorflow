# TFLite Android Model Benchmark Tool

## Description

This Android benchmark app is a simple wrapper around the TensorFlow Lite
[command-line benchmark utility](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

Pushing and executing binaries directly on Android is a valid approach to
benchmarking, but it can result in subtle (but observable) differences in
performance relative to execution within an actual Android app. In particular,
Android's scheduler tailors behavior based on thread and process priorities,
which differ between a foreground Activity/Application and a regular background
binary executed via `adb shell ...`. This tailored behavior is most evident when
enabling multi-threaded CPU execution with TensorFlow Lite.

To that end, this app offers perhaps a more faithful view of runtime performance
that developers can expected when deploying TensorFlow Lite with their
application.

## To build/install/run

(0) Refer to
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
to edit the `WORKSPACE` to configure the android NDK/SDK.

(1) Build for your specific platform, e.g.:

```
bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++11' \
  tensorflow/lite/tools/benchmark/android:benchmark_model
```

(2) Connect your phone. Install the benchmark APK to your phone with adb:

```
adb install -r -d bazel-bin/tensorflow/lite/tools/benchmark/android/benchmark_model.apk
```

(3) Push the compute graph that you need to test.

```
adb push mobilenet_quant_v1_224.tflite /data/local/tmp
```

(4) Run the benchmark. Additional command-line flags are documented
[here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/README.md)
and can be appended to the `args` string alongside the required `--graph` flag
(note that all args must be nested in the single quoted string that follows the
args key).

```
adb shell am start -S -n \
  org.tensorflow.lite.benchmark/org.tensorflow.lite.benchmark.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/mobilenet_quant_v1_224.tflite --num_threads=4"'
```

(5) The results will be available in Android's logcat, e.g.:

```
adb logcat | grep "Average inference"

... tflite  : Average inference timings in us: Warmup: 91471, Init: 4108, Inference: 80660.1
```
