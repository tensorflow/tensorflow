# TFLite Model Benchmark Tool

## Description

A simple C++ binary to benchmark a TFLite model and its individual operators,
both on desktop machines and on Android. The binary takes a TFLite model,
generates random inputs and then repeatedly runs the model for specified number
of runs. Aggregate latency statistics are reported after running the benchmark.

The instructions below are for running the binary on Desktop and Android,
for iOS please use the
[iOS benchmark app](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios).

An experimental Android APK wrapper for the benchmark model utility offers more
faithful execution behavior on Android (via a foreground Activity). It is
located
[here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android).

## Parameters

The binary takes the following required parameters:

*   `graph`: `string` \
    The path to the TFLite model file.

and the following optional parameters:

*   `num_threads`: `int` (default=1) \
    The number of threads to use for running TFLite interpreter.
*   `warmup_runs`: `int` (default=1) \
    The number of warmup runs to do before starting the benchmark.
*   `num_runs`: `int` (default=50) \
    The number of runs. Increase this to reduce variance.
*   `run_delay`: `float` (default=-1.0) \
    The delay in seconds between subsequent benchmark runs. Non-positive values
    mean use no delay.
*   `use_hexagon`: `bool` (default=false) \
    Whether to use the Hexagon delegate. Not all devices may support the Hexagon
    delegate, refer to the TensorFlow Lite documentation for more information
    about which devices/chipsets are supported and about how to get the
    required libraries. To use the Hexagon delegate also build the
    hexagon_nn:libhexagon_interface.so target and copy the library to the
    device. All libraries should be copied to /data/local/tmp on the device.
*   `use_nnapi`: `bool` (default=false) \
    Whether to use [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/).
    This API is available on recent Android devices. Note that some Android P
    devices will fail to use NNAPI for models in `/data/local/tmp/` and this
    benchmark tool will not correctly use NNAPI. When on Android Q+, will also
    print the names of NNAPI accelerators accessible through the
    `nnapi_accelerator_name` flag.
*   `nnapi_accelerator_name`: `str` (default="") \
    The name of the NNAPI accelerator to use (requires Android Q+). If left
    blank, NNAPI will automatically select which of the available accelerators
    to use.
*   `nnapi_execution_preference`: `string` (default="") \
    Which [NNAPI execution preference](https://developer.android.com/ndk/reference/group/neural-networks.html#group___neural_networks_1gga034380829226e2d980b2a7e63c992f18af727c25f1e2d8dcc693c477aef4ea5f5)
    to use when executing using NNAPI. Should be one of the
    following: fast_single_answer, sustained_speed, low_power, undefined.
*   `use_legacy_nnapi`: `bool` (default=false) \
    Whether to use the legacy
    [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/)
    TFLite path, which requires the graph to be fully compatible with NNAPI.
    This is available on recent Android devices. Note that some Android P
    devices will fail to use NNAPI for models in `/data/local/tmp/` and this
    benchmark tool will not correctly use NNAPI.
*   `use_gpu`: `bool` (default=false) \
    Whether to use the [GPU accelerator delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu).
    This option is currently only available on Android and iOS devices.
*   `gpu_wait_type`: `str` (default="") \
    Which GPU wait_type option to use, when using GPU delegate on iOS. Should be
    one of the following: passive, active, do_not_wait, aggressive. When left
    blank, passive mode is used by default.
*   `enable_op_profiling`: `bool` (default=false) \
    Whether to enable per-operator profiling measurement.

## To build/install/run

### On Android:

(0) Refer to https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android to edit the `WORKSPACE` to configure the android NDK/SDK.

(1) Build for your specific platform, e.g.:

```
bazel build -c opt \
  --config=android_arm \
  tensorflow/lite/tools/benchmark:benchmark_model
```

(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):

```
adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
```

(3) Make the binary executable.

```
adb shell chmod +x /data/local/tmp/benchmark_model
```

(4) Push the compute graph that you need to test. For example:

```
adb push mobilenet_quant_v1_224.tflite /data/local/tmp
```

(5) Optionally, install Hexagon libraries on device.

That step is only needed when using the Hexagon delegate.

```
bazel build --config=android_arm \
  tensorflow/lite/experimental/delegates/hexagon/hexagon_nn:libhexagon_interface.so
adb push bazel-bin/tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/libhexagon_interface.so /data/local/tmp
adb push libhexagon_nn_skel*.so /data/local/tmp
```

(6) Run the benchmark. For example:

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=4
```

### On desktop:
(1) build the binary

```
bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model
```

(2) Run on your compute graph, similar to the Android case but without the need of adb shell.
For example:

```
bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=mobilenet_quant_v1_224.tflite \
  --num_threads=4
```

The MobileNet graph used as an example here may be downloaded from [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip).


## Reducing variance between runs on Android.

Most modern Android phones use [ARM big.LITTLE](https://en.wikipedia.org/wiki/ARM_big.LITTLE)
architecture where some cores are more power hungry but faster than other cores.
When running benchmarks on these phones there can be significant variance
between different runs of the benchmark. One way to reduce variance between runs
is to set the [CPU affinity](https://en.wikipedia.org/wiki/Processor_affinity)
before running the benchmark. On Android this can be done using the `taskset`
command.
E.g. for running the benchmark on big cores on Pixel 2 with a single thread one
can use the following command:

```
adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=1
```

where `f0` is the affinity mask for big cores on Pixel 2.
Note: The affinity mask varies with the device.

## Profiling model operators
The benchmark model binary also allows you to profile operators and give
execution times of each operator. To do this, pass the flag
`--enable_op_profiling=true` to `benchmark_model` during invocation, e.g.,

```
adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --enable_op_profiling=true
```

When enabled, the `benchmark_model` binary will produce detailed statistics for
each operation similar to those shown below:

```

============================== Run Order ==============================
	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 CONV_2D	    0.000	    4.269	    4.269	  0.107%	  0.107%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_0/Relu6]
	       DEPTHWISE_CONV_2D	    4.270	    2.150	    2.150	  0.054%	  0.161%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6]
	                 CONV_2D	    6.421	    6.107	    6.107	  0.153%	  0.314%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   12.528	    1.366	    1.366	  0.034%	  0.348%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6]
	                 CONV_2D	   13.895	    4.195	    4.195	  0.105%	  0.454%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   18.091	    1.260	    1.260	  0.032%	  0.485%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6]
	                 CONV_2D	   19.352	    6.652	    6.652	  0.167%	  0.652%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   26.005	    0.698	    0.698	  0.018%	  0.670%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6]
	                 CONV_2D	   26.703	    3.344	    3.344	  0.084%	  0.754%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   30.047	    0.646	    0.646	  0.016%	  0.770%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6]
	                 CONV_2D	   30.694	    5.800	    5.800	  0.145%	  0.915%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   36.495	    0.331	    0.331	  0.008%	  0.924%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6]
	                 CONV_2D	   36.826	    2.838	    2.838	  0.071%	  0.995%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   39.665	    0.439	    0.439	  0.011%	  1.006%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6]
	                 CONV_2D	   40.105	    5.293	    5.293	  0.133%	  1.139%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   45.399	    0.352	    0.352	  0.009%	  1.147%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6]
	                 CONV_2D	   45.752	    5.322	    5.322	  0.133%	  1.281%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   51.075	    0.357	    0.357	  0.009%	  1.290%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6]
	                 CONV_2D	   51.432	    5.693	    5.693	  0.143%	  1.433%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   57.126	    0.366	    0.366	  0.009%	  1.442%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6]
	                 CONV_2D	   57.493	    5.472	    5.472	  0.137%	  1.579%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   62.966	    0.364	    0.364	  0.009%	  1.588%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6]
	                 CONV_2D	   63.330	    5.404	    5.404	  0.136%	  1.724%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   68.735	    0.155	    0.155	  0.004%	  1.728%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6]
	                 CONV_2D	   68.891	    2.970	    2.970	  0.074%	  1.802%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   71.862	    0.206	    0.206	  0.005%	  1.807%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6]
	                 CONV_2D	   72.069	    5.888	    5.888	  0.148%	  1.955%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6]
	         AVERAGE_POOL_2D	   77.958	    0.036	    0.036	  0.001%	  1.956%	     0.000	        0	[MobilenetV1/Logits/AvgPool_1a/AvgPool]
	                 CONV_2D	   77.994	    1.445	    1.445	  0.036%	  1.992%	     0.000	        0	[MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd]
	                 RESHAPE	   79.440	    0.002	    0.002	  0.000%	  1.992%	     0.000	        0	[MobilenetV1/Predictions/Reshape]
	                 SOFTMAX	   79.443	    0.029	    0.029	  0.001%	  1.993%	     0.000	        0	[MobilenetV1/Predictions/Softmax]

============================== Top by Computation Time ==============================
	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 CONV_2D	   19.352	    6.652	    6.652	  0.167%	  0.167%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6]
	                 CONV_2D	    6.421	    6.107	    6.107	  0.153%	  0.320%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6]
	                 CONV_2D	   72.069	    5.888	    5.888	  0.148%	  0.468%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6]
	                 CONV_2D	   30.694	    5.800	    5.800	  0.145%	  0.613%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6]
	                 CONV_2D	   51.432	    5.693	    5.693	  0.143%	  0.756%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
	                 CONV_2D	   57.493	    5.472	    5.472	  0.137%	  0.893%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6]
	                 CONV_2D	   63.330	    5.404	    5.404	  0.136%	  1.029%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6]
	                 CONV_2D	   45.752	    5.322	    5.322	  0.133%	  1.162%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
	                 CONV_2D	   40.105	    5.293	    5.293	  0.133%	  1.295%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
	                 CONV_2D	    0.000	    4.269	    4.269	  0.107%	  1.402%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_0/Relu6]

Number of nodes executed: 31
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	                 CONV_2D	       15	     1.406	    89.270%	    89.270%	     0.000	        0
	       DEPTHWISE_CONV_2D	       13	     0.169	    10.730%	   100.000%	     0.000	        0
	                 SOFTMAX	        1	     0.000	     0.000%	   100.000%	     0.000	        0
	                 RESHAPE	        1	     0.000	     0.000%	   100.000%	     0.000	        0
	         AVERAGE_POOL_2D	        1	     0.000	     0.000%	   100.000%	     0.000	        0

Timings (microseconds): count=50 first=79449 curr=81350 min=77385 max=88213 avg=79732 std=1929
Memory (bytes): count=0
31 nodes observed


Average inference timings in us: Warmup: 83235, Init: 38467, Inference: 79760.9
```

## Benchmark multiple performance options in a single run

A convenient and simple C++ binary is also provided to benchmark multiple
performance options in a single run. This binary is built based on the
aforementioned benchmark tool that could only benchmark a single performance
option at a time. They share the same build/install/run process, but the BUILD
target name of this binary is `benchmark_model_performance_options` and it takes
some additional parameters as detailed below.

### Additional Parameters
*   `perf_options_list`: `string` (default='all') \
    A comma-separated list of TFLite performance options to benchmark.
*   `option_benchmark_run_delay`: `float` (default=-1.0) \
    The delay between two consecutive runs of benchmarking performance options
    in seconds.
*   `random_shuffle_benchmark_runs`: `bool` (default=true) \
    Whether to perform all benchmark runs, each of which has different
    performance options, in a random order.
