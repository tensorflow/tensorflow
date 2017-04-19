# TensorFlow Model Benchmark Tool

## Description

A simple C++ binary to benchmark a compute graph and its individual operators,
both on desktop machines and on Android.

## To build/install/run

### On Android:

(1) build for your specific platform, e.g.:
```bash
$bazel build -c opt \
  --crosstool_top=//external:android/crosstool \
  --cpu=armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  tensorflow/tools/benchmark:benchmark_model
```

(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):
```bash
$adb push bazel-bin/tensorflow/tools/benchmark/benchmark_model /data/local/tmp
```

(3) Push the compute graph that you need to test. For example:
     adb push tensorflow_inception_graph.pb /data/local/tmp

(4) Run the benchmark. For example:
```bash
$adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"
```
### On desktop:
(1) build the binary
```bash
$bazel build --config opt tensorflow/tools/benchmark:benchmark_model
```

(2) Run on your compute graph, similar to the Android case but without the need of adb shell.
For example:
```bash
$bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"
```

The Inception graph used as an example here may be downloaded from
https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

## To visualize benchmark

Since [timeline](https://github.com/tensorflow/tensorflow/blob/27711108b5fce2e1692f9440631a183b3808fa01/tensorflow/python/client/timeline.py) is a python only visualization tool, we need to write collected **StepStats** to a file and then use timeline to generate JSON-formatted file in Chrome Trace format.

To store collected **StepStats**, set `benchmark_model`' s `--step_stats_name` flag to the name of the file you want to write to.

To generate Chrome Trace format, run `transform_chrome_trace.py` directly with **StepStats** file and desired JSON file as arguments. Navigate to `chrome://tracing` to visualize the benchmark.
