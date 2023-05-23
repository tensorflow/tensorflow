# TensorFlow Model Benchmark Tool

## Description

A simple C++ binary to benchmark a compute graph and its individual operators,
both on desktop machines and on Android.

## To build/install/run

### On Android:

(0) Refer to https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
to edit the `WORKSPACE` to configure the android NDK/SDK.

(1) build for your specific platform, e.g.:

```
bazel build -c opt \
  --crosstool_top=//external:android/crosstool \
  --cpu=armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --config monolithic \
  tensorflow/tools/benchmark:benchmark_model
```

(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):

```
adb push bazel-bin/tensorflow/tools/benchmark/benchmark_model /data/local/tmp
```

(3) Push the compute graph that you need to test. For example:

```
adb push tensorflow_inception_graph.pb /data/local/tmp
```

(4) Run the benchmark. For example:

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"
```

### On desktop:
(1) build the binary

```
bazel build -c opt tensorflow/tools/benchmark:benchmark_model
```

(2) Run on your compute graph, similar to the Android case but without the need
of adb shell. For example:

```
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"
```

The Inception graph used as an example here may be downloaded from
https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

## Model downloader
To download TF .pb graphs of several popular models, run:

```sh
bash download_models.sh
```

## Comparing performance with vanilla TF

We provide example scripts comparing TF-oneDNN performance with vanilla TF's
that users can modify for their own benchmarks. The scripts assume that models
are already downloaded by `download_models.sh`. To run end-to-end model
performance comparison between TF-oneDNN and vanilla TF, call

```sh
bash download_models.sh  # Skip this step if models are already downloaded.
bash run_onednn_benchmarks.sh
```

The output is a summary table in a CSV file: results.csv. Example output:

```sh
Showing runtimes in microseconds. `?` means not available.
               Model,  Batch,        Vanilla,         oneDNN,    Speedup
          bert-large,      1,              x,              y,        x/y
          bert-large,     16,            ...,            ...,        ...
          bert-large,     64,            ...,            ...,        ...          
           inception,      1,            ...,            ...,        ...
           inception,     16,            ...,            ...,        ...
           inception,     64,            ...,            ...,        ...           
                                        ⋮
        ssd-resnet34,      1,              ?,            ...,          ?
        ssd-resnet34,     16,              ?,            ...,          ?
        ssd-resnet34,     64,              ?,            ...,          ?        
```

Vanilla TF can't run `ssd-resnet34` on CPU because it doesn't support NCHW
format.
