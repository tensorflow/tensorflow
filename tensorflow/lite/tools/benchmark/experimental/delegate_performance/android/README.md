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

To those ends, this app offers a more faithful view of runtime performance
(accuracy and latency) that developers can expect when deploying TensorFlow Lite
with their application, and the app provides a single entrypoint to various
performance metrics to avoid the need to switch between different benchmarking
apps.

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

##### Options

-   `tflite_settings_files`: `str` (required) the comma-delimited paths to the
    JSON-encoded delegate `TFLiteSettings` file(s), which is defined in [configuration.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/acceleration/configuration/configuration.proto).
-   Additional optional command-line flags are documented
    [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/README.md)
    and can be appended to the `args` string (note that all args must be nested
    in the single quoted string that follows the args key).

Currently the tool uses a 'mobilenet_v1_1.0_224.tflite' and
'mobilenet_quant_v1_224.tflite' model for latency and accuracy benchmarking. The
TF Lite model files are bundled into the app during the build process. We plan
to expand the number of models in the future.

1.  Dump the test target delegate settings file on device. Example command:

    ```
    adb shell 'echo "{
    \"delegate\": XNNPACK
    }
    "> /data/local/tmp/xnnpack_settings.json'
    ```

1.  Run the benchmark by supplying the settings file via the required
    `--tflite_settings_files` flag.

    ```
    adb shell "am start -S \
    -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkLatencyActivity \
    --esa --tflite_settings_files '/data/local/tmp/xnnpack_settings.json'"
    ```

1.  The results will be available in Android logcat as well as the app's file
    directory, e.g.:

    ```
    adb logcat | grep "Inference timings in us"

    ... tflite  : Inference timings in us: Init: 1007529, First inference: 4098, Warmup (avg): 1686.59, Inference (avg): 1687.92
    ```

    The tool also shows overall results.

    ```
    adb logcat | grep 'Latency benchmark result'
    ```

    which might show output like the following.

    ```
    01-13 15:00:04.407 24897 24897 I TfLiteLatencyImpl: Latency benchmark result for /data/local/tmp/xnnpack_settings.json: PASS
    ```

    To obtain the generated report run.  Note that in order for this command to
    succeed the APK needs to be debuggable (e.g. built with `-c dbg`).

    ```
    adb shell run-as org.tensorflow.lite.benchmark.delegateperformance "cat /data/user/0/org.tensorflow.lite.benchmark.delegateperformance/files/delegate_performance_result/latency/mobilenet_v1_1.0_224.csv"
    ```

    This might produce output like the following:

    ```
    Metric,0 (default_delegate),4 (/data/local/tmp/xnnpack_settings.json),%
    inference_latency_average_us,93703.56,93609.74,-0.1%
    initialization_memory_total_allocated_mebibyte,0.0,0.0,0%
    overall_memory_max_rss_mebibyte,42.191406,0.0,-100.0%
    model_size_megabyte,-1.0E-6,-1.0E-6,0%
    initialization_latency_us,43174.0,39517.0,-8.5%
    warmup_latency_standard_deviation,1918.0,1524.0,-20.5%
    initialization_memory_in_use_mebibyte,21.564835,21.541794,-0.1%
    overall_memory_in_use_mebibyte,23.624878,23.622772,-0.0%
    warmup_latency_max_us,98737.0,97734.0,-1.0%
    warmup_latency_min_us,93500.0,93415.0,-0.1%
    inference_latency_min_us,93312.0,93228.0,-0.1%
    overall_memory_total_allocated_mebibyte,0.0,0.0,0%
    inference_latency_max_us,94186.0,94942.0,0.8%
    initialization_memory_max_rss_mebibyte,34.625,0.0,-100.0%
    warmup_latency_average_us,94449.836,94348.664,-0.1%
    inference_latency_standard_deviation,161.0,251.0,55.9%
    ```

#### Accuracy benchmarking

##### Options

-   `tflite_settings_files`: `str` (required) the comma-delimited paths to the
    JSON-encoded delegate `TFLiteSettings` file(s), which is defined in [configuration.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/acceleration/configuration/configuration.proto).

##### Steps

1.  Dump the test target delegate settings file on device. Example command:

    ```
    adb shell 'echo "{
    \"delegate\": XNNPACK
    }
    "> /data/local/tmp/xnnpack_settings.json'
    ```

1.  Run the benchmark. Currently the accuracy benchmark only supports parsing
    arguments for delegate selection.

    Run the test with the XNNPack delegate (default):

    ```
    adb shell "am start -S \
    -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkAccuracyActivity \
    --esa --tflite_settings_files '/data/local/tmp/xnnpack_settings.json'"
    ```

1.  The results will be available in Android logcat, e.g.:

    TODO(b/250877013): improve performance thresholding and result reporting

    ```
    adb logcat | grep "tflite"

    ... tflite  : tflite  :   accuracy: ok
    ```

#### Benchmarking with stable delegates

The stable delegate provider dynamically loads a stable delegate symbol from the
provided binary (shared object) file. In order to use Delegate Performance
Benchmark with a stable delegate, users will need to push the shared object file
to the file directory of Delegate Performance Benchmark:
`/data/data/org.tensorflow.lite.benchmark.delegateperformance/files/`.

Example steps to start the latency benchmark with a stable delegate:

1.  Build and push the example stable delegate binary that you want to test.
    Here we use
    [the sample stable delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate)
    as an example.

    ```
    # Make sure the app is debuggable to allow copying the .so file to the app's
    # file directory.
    bazel build -c dbg \
    --config=android_arm64 \
    tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate

    # Set the permissions so that we can overwrite a previously installed delegate
    chmod 755 bazel-bin/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so

    # Ensure the delegateperformance files path exists
    adb shell run-as org.tensorflow.lite.benchmark.delegateperformance mkdir -p /data/data/org.tensorflow.lite.benchmark.delegateperformance/files

    # Install the sample delegate.
    adb push \
    bazel-bin/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so \
    /data/local/tmp/
    adb shell run-as org.tensorflow.lite.benchmark.delegateperformance \
    cp /data/local/tmp/libtensorflowlite_sample_stable_delegate.so \
      /data/data/org.tensorflow.lite.benchmark.delegateperformance/files/
    ```

1. Dump the test sample delegate settings file on device. Example command:

    ```
    adb shell 'echo "{
    \"stable_delegate_loader_settings\": {
      \"delegate_path\": \"/data/data/org.tensorflow.lite.benchmark.delegateperformance/files/libtensorflowlite_sample_stable_delegate.so\"
    }
    // Add concrete delegate settings for the test target delegate.
    }
    "> /data/local/tmp/stable_delegate_settings.json'
    ```

##### Latency Benchmarking

```
adb shell "am start -S \
  -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkLatencyActivity \
  --esa --tflite_settings_files '/data/local/tmp/stable_delegate_settings.json'"
```

##### Accuracy Benchmarking

```
adb shell "am start -S \
  -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkAccuracyActivity \
  --esa --tflite_settings_files '/data/local/tmp/stable_delegate_settings.json'"
```
