# TensorFlow Lite Delegate Performance Benchmark (Android APK)

## Description

This Android Delegate Performance Benchmark (DPB) app is a simple wrapper around
the TensorFlow Lite
[benchmark tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
and
[MiniBenchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/acceleration/mini_benchmark)
with the focus on testing Tensorflow Lite delegates that implement stable
delegate ABI.

Development of TensorFlow Lite delegates needs both accuracy and latency
evaluations for catching potential performance regressions. Pushing and
executing both latency and accuracy testing binaries directly on an Android
device is a valid approach to benchmarking, but it can result in subtle (but
observable) differences in performance relative to execution within an actual
Android app. In particular, Android's scheduler tailors behavior based on thread
and process priorities, which differ between a foreground Activity/Application
and a regular background binary executed via `adb shell ...`.

In addition to that, having multiple benchmarking apps for different performance
metric evaluations could potentially cost development effort unnecessarily.

To those ends, this app offers a more faithful view of runtime performance
(accuracy and latency) that developers can expect when using TensorFlow Lite
delegates with Android apps, and the app provides a single entrypoint to
various performance metrics to avoid the need to switch between different
benchmarking apps.

## To build/install/run

### Build
1.  Clone the TensorFlow repo with

    ```
    git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
    ```

    Note: --recurse-submodules is necessary to prevent some issues with protobuf
    compilation.

1.  Refer to
    [this page](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_without_docker)
    for setting up a development environment. Although there are several
    practical tips:

    -   When installing Bazel, for Ubuntu Linux, `sudo apt update && sudo apt
        install bazel` may be the easiest way. However sometimes you may need
        `sudo apt update && sudo apt install bazel-5.3.0` if prompted.
    -   When installing Android NDK and SDK, using Android Studio's SDK Manager
        may be the easiest way.
    -   The versions which we have verified are working:
        -   Android NDK API level: 21
        -   Android SDK API level: 33
        -   Android build tools version: 30.0.0
    -   Run the `./configure` script in the root TensorFlow checkout directory,
        and answer "Yes" when the script asks to interactively configure the
        `./WORKSPACE` for Android builds.

1.  Build for your specific platform, e.g.:

    ```
    bazel build -c opt \
      --config=android_arm64 \
      tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:delegate_performance_benchmark
    ```

### Install

1.  Connect to a physical device. Install the benchmark APK with adb:

    ```
    adb install -r -d -g bazel-bin/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/delegate_performance_benchmark.apk
    ```

    Note: Make sure to install with "-g" option to grant the permission for
    reading external storage.

### Run

#### Benchmarking with stable delegates

The delegate-under-test must implement the
[stable_delegate_interface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h)
API. The stable delegate provider dynamically loads stable delegate symbols
from the provided binary (shared object) file. In order to use Delegate
Performance Benchmark with a stable delegate, you would need to push the shared
object file to the file directory of Delegate Performance Benchmark:
`/data/data/org.tensorflow.lite.benchmark.delegateperformance/files/`.


1.  Build and push the stable delegate binary that you want to test.
    Here we use the
    [sample stable delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate)
    as an example.

    ```
    bazel build -c opt \
      --config=android_arm64 \
      tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate

    # Set the permissions so that we can overwrite a previously installed delegate.
    chmod 755 bazel-bin/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so

    # Ensure the delegateperformance files path exists.
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

#### Supported models

Currently DPB uses a `mobilenet_v1_1.0_224.tflite` and
`mobilenet_quant_v1_224.tflite` model for latency and accuracy benchmarking. The
TF Lite model files are bundled into the app during the build process. We plan
to expand the supported models based on future use cases.

Note: The sample stable delegate provided here only supports ADD and SUB
operations thus aforementioned mobilenet models would not actually be delegated.
To test your own delegate against the models, please update
`stable_delegate_loader_settings` with your delegate path.  To get feedback
early in the development process, e.g. while working towards supporting more
OPs, you can run the `benchmark_model` tool, which supports stable delegates and
can be supplied with arbitrary models via the `--graph` CLI parameter. See [this document](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/README.md#tf-lite-benchmark-tool)
which shows how to run a model with ADD operations through the sample stable
delegate.

#### Latency benchmarking

##### Options

-   `tflite_settings_files`: `str` (required) the comma-delimited paths to the
    JSON-encoded delegate `TFLiteSettings` file(s), which is defined in [configuration.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/acceleration/configuration/configuration.proto).
-   Additional optional command-line flags are documented
    [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/README.md)
    and can be appended to the `args` string (note that all args must be nested
    in the single quoted string that follows the args key).

##### Steps

1.  Run the latency benchmark by supplying the settings file via the required
    `--tflite_settings_files` flag.

    ```
    adb shell "am start -S \
  -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkLatencyActivity \
  --esa --tflite_settings_files '/data/local/tmp/stable_delegate_settings.json'"
    ```

1.  The results will be available in Android logcat as well as the app's file
    directory, e.g.:

    ```
    adb logcat -c && adb logcat -v color  | grep "Inference timings in us"

    ... tflite  : Inference timings in us: Init: 5811, First inference: 67743, Warmup (avg): 65539, Inference (avg): 65155.5
    ```

    The tool also shows overall results.

    ```
    adb logcat -c && adb logcat -v color  | grep 'Latency benchmark result'
    ```

    which might show output like the following.

    ```
    ... TfLiteLatencyImpl: Latency benchmark result for /data/local/tmp/stable_delegate_settings.json: PASS
    ```

    For a summarized view, run

    ```
    adb shell run-as org.tensorflow.lite.benchmark.delegateperformance "cat /data/user/0/org.tensorflow.lite.benchmark.delegateperformance/files/delegate_performance_result/latency/mobilenet_v1_1.0_224.csv" | column -t -s,
    ```

    It would produce output like the following:

    |Metric|0 (default_delegate)|0 (/data/local/tmp/stable_delegate_settings.json)|%|
    |---|---|---|---|
    |inference_latency_average_us|97936.44|97963.66|0.0%|
    |initialization_memory_total_allocated_mebibyte|0.0|0.0|0%|
    |overall_memory_max_rss_mebibyte|43.246094|0.65625|-98.5%|
    |model_size_megabyte|-1.0E-6|-1.0E-6|0%|
    |initialization_latency_us|39640.0|40155.0|1.3%|
    |warmup_latency_standard_deviation|1644.0|1463.0|-11.0%|
    |initialization_memory_in_use_mebibyte|23.091217|27.671188|19.8%|
    |overall_memory_in_use_mebibyte|25.15126|29.752167|18.3%|
    |warmup_latency_max_us|102423.0|101935.0|-0.5%|
    |warmup_latency_min_us|97855.0|97868.0|0.0%|
    |inference_latency_min_us|97597.0|97657.0|0.1%|
    |overall_memory_total_allocated_mebibyte|0.0|0.0|0%|
    |inference_latency_max_us|98295.0|98370.0|0.1%|
    |initialization_memory_max_rss_mebibyte|34.39453|0.0|-100.0%|
    |warmup_latency_average_us|98752.836|98672.164|-0.1%|
    |inference_latency_standard_deviation|146.0|143.0|-2.1%|

#### Accuracy benchmarking

##### Options

-   `tflite_settings_files`: `str` (required) the comma-delimited paths to the
    JSON-encoded delegate `TFLiteSettings` file(s), which is defined in [configuration.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/acceleration/configuration/configuration.proto).

##### Steps

1.  Run the accuracy benchmark by supplying the settings file via the required
    `--tflite_settings_files` flag.

    ```
    adb shell "am start -S \
  -n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkAccuracyActivity \
  --esa --tflite_settings_files '/data/local/tmp/stable_delegate_settings.json'"
    ```

1.  The results will be available in Android logcat, e.g.:

    ```
    adb logcat -c && adb logcat -v color | grep "tflite"

    ... tflite  : tflite  :   accuracy: ok
    ```

    For a summarized view, run
    
    ```
    adb shell run-as org.tensorflow.lite.benchmark.delegateperformance "cat /data/user/0/org.tensorflow.lite.benchmark.delegateperformance/files/delegate_performance_result/accuracy/mobilenet_v1_1.0_224_with_validation.csv" | column -t -s,
    ```

    It would produce output like the following:

    |Metric|0 (default_delegate)|0 (/data/local/tmp/stable_delegate_settings.json)|%|
    |---|---|---|---|
    |max_memory_kb|0.0|0.0|0%|
    |symmetric_kl_divergence(average)|3.5098449E-9|3.5098449E-9|0%|
    |mse(average)|1.0153732E-16|1.0153732E-16|0%|
    |ok|1.0|1.0|0%|
