# TensorFlow Lite Delegate Performance Benchmark with Android Apk

## Description

This Android Delegate Performance Benchmark app is a simple wrapper around the
TensorFlow Lite
[benchmark tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
and
[MiniBenchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/acceleration/mini_benchmark).

Development against TensorFlow Lite needs both accuracy and latency evaluations
for catching potential performance regressions. Pushing and executing both
latency and accuracy testing binaries directly on an Android device is a valid
approach to benchmarking, but it can result in subtle (but observable)
differences in performance relative to execution within an actual Android app.
In particular, Android's scheduler tailors behavior based on thread and process
priorities, which differ between a foreground Activity/Application and a regular
background binary executed via `adb shell ...`.

In addition to that, having multiple benchmarking apps for different performance
metric evaluations could potentially cost development effort unnecessarily.

To that end, this app offers a more faithful view of runtime performance
(accuracy and latency) that developers can expect when deploying TensorFlow Lite
with their application.

## To build/install/run

### Build

1.  Refer to
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/android/test
    to edit the `WORKSPACE` to configure the android NDK/SDK.

1.  Build for your specific platform, e.g.:

```
bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:delegate_performance_benchmark
```

### Install

1.  Connect your phone. Install the benchmark APK on your phone with adb:

```
adb install -r -d -g bazel-bin/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/delegate_performance_benchmark.apk
```

Note: Make sure to install with "-g" option to grant the permission for reading
external storage.

### Run

#### Latency benchmarking

1.  Push the TensorFlow Lite model that you need to test.

```
adb push mobilenet_quant_v1_224.tflite /data/local/tmp
```

1.  Run the benchmark. Additional command-line flags are documented
    [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/README.md)
    and can be appended to the `args` string alongside the required `--graph`
    flag (note that all args must be nested in the single quoted string that
    follows the args key). {value=2}

```
adb shell "am start -S \
  -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkLatencyActivity \
  --esa --args '--graph=/data/local/tmp/mobilenet_quant_v1_224.tflite, \
               --num_threads=4'"
```

1.  The results will be available in Android logcat as well as the app's file
    directory, e.g.: {value=3}

```
adb logcat | grep "Inference timings in us"

... tflite  : Inference timings in us: Init: 1007529, First inference: 4098, Warmup (avg): 1686.59, Inference (avg): 1687.92
```

and `report.json`

```
{
  "name": "TFLite benchmark",
  "status": "OK",
  "average time in us": "init: 1007529, warmup: 1686.59, inference: 1687.92",
  ...
}
```

#### Accuracy benchmarking

TODO(b/240438534): Add mini-benchmark to the app
