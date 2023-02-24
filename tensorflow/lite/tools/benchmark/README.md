# TFLite Model Benchmark Tool with C++ Binary

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

*   `num_threads`: `int` (default=-1) \
    The number of threads to use for running TFLite interpreter. By default,
    this is set to the platform default value -1.
*   `warmup_runs`: `int` (default=1) \
    The number of warmup runs to do before starting the benchmark.
*   `num_runs`: `int` (default=50) \
    The number of runs. Increase this to reduce variance.
*   `max_secs` : float (default=150.0) \
    The maximum number of seconds the benchmark will run before being
    terminated.
*   `run_delay`: `float` (default=-1.0) \
    The delay in seconds between subsequent benchmark runs. Non-positive values
    mean use no delay.
*   `run_frequency`: `float` (default=-1.0) \
    The frequency of running a benchmark run as the number of prorated runs per
    second. If the targeted rate per second cannot be reached, the benchmark
    would start the next run immediately, trying its best to catch up. If set,
    this will override the `run_delay` parameter. A non-positive value means
    there is no delay between subsequent runs.
*   `enable_op_profiling`: `bool` (default=false) \
    Whether to enable per-operator profiling measurement.
*   `max_profiling_buffer_entries`: `int` (default=1024) \
    The initial max number of profiling events that will be stored during each
    inference run. It is only meaningful when `enable_op_profiling` is set to
    `true`. Note, the actual value of this parameter will be adjusted if the
    model has more nodes than the specified value of this parameter. Also, when
    `allow_dynamic_profiling_buffer_increase` is set to `true`, the number of
    profiling buffer entries will be increased dynamically.
*   `allow_dynamic_profiling_buffer_increase`: `bool` (default=false) \
    Whether allowing dynamic increase on the number of profiling buffer entries.
    It is only meaningful when `enable_op_profiling` is set to `true`. Note,
    allowing dynamic buffer size increase may cause more profiling overhead,
    thus it is preferred to set `max_profiling_buffer_entries` to a large-enough
    value.

*   `profiling_output_csv_file`: `str` (default="") \
    File path to export profile data to as CSV. The results are printed to
    `stdout` if option is not set. Requires `enable_op_profiling` to be `true`
    and the path to include the name of the output CSV; otherwise results are
    printed to `stdout`.

*   `print_preinvoke_state`: `bool` (default=false) \
    Whether to print out the TfLite interpreter internals just before calling
    tflite::Interpreter::Invoke. The internals will include allocated memory
    size of each tensor etc. Enabling this could help understand TfLite graph
    and memory usage.

*   `print_postinvoke_state`: `bool` (default=false) \
    Whether to print out the TfLite interpreter internals just before benchmark
    completes (i.e. after all repeated Invoke calls complete). The internals
    will include allocated memory size of each tensor etc. Enabling this could
    help understand TfLite graph and memory usage, particularly when there are
    dynamic-shaped tensors in the graph.

*   `report_peak_memory_footprint`: `bool` (default=false) \
    Whether to report the peak memory footprint by periodically checking the
    memory footprint. Internally, a separate thread will be spawned for this
    periodic check. Therefore, the performance benchmark result could be
    affected.

*   `memory_footprint_check_interval_ms`: `int` (default=50) \
    The interval in millisecond between two consecutive memory footprint checks.
    This is only used when --report_peak_memory_footprint is set to true.

*   `dry_run`: `bool` (default=false) \
    Whether to run the tool just with simply loading the model, allocating
    tensors etc. but without actually invoking any op kernels.

*   `verbose`: `bool` (default=false) \
    Whether to log parameters whose values are not set. By default, only log
    those parameters that are set by parsing their values from the commandline
    flags.

*   `release_dynamic_tensors`: `bool` (default=false) \
    Whether to configure the Interpreter to immediately release the memory of
    dynamic tensors in the graph once they are not used.

*   `optimize_memory_for_large_tensors`: `int` (default=0) \
    Whether to optimize memory usage for large tensors with sacrificing latency.
    When the feature is enabled, `release_dynamic_tensors` is also enabled.

This list of parameters is not exhaustive. See
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc)
and
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_tflite_model.cc)
for all parameters that the binary takes.

### Model input parameters
By default, the tool will use randomized data for model inputs. The following
parameters allow users to specify customized input values to the model when
running the benchmark tool:

*   `input_layer`: `string` \
    A comma-separated list of input layer names, e.g. 'input1,input2'. Note all
    inputs of the model graph need to be specified. However, the input name
    does not need to match that encoded in the model. Additionally, the order
    of input layer names specified here is assumed to be same with that is seen
    by the Tensorflow Lite interpreter. This is a bit inconvenient but the
    [visualization tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py)
    should help to find this order.
*   `input_layer_shape`: `string` \
    A colon-separated list of input layer shapes, where each shape is a
    comma-separated list, e.g. '1,30:1,10'. Similar to `input_layer`, this
    parameter also requires shapes of all inputs be specified, and the order of
    inputs be same with that is seen by the interpreter.
*   `input_layer_value_range`: `string` \
    A map-like string representing value range for *integer* input layers. Each
    item is separated by ':', and the item value consists of input layer name
    and integer-only range values (both low and high are inclusive) separated by
    ',', e.g. 'input1,1,2:input2,0,254'. Note that the input layer name must
    exist in the list of names specified by `input_layer`.
*   `input_layer_value_files`: `string` \
    A map-like string representing files that contain input values. Each
    item is separated by ',', and the item value consists of input layer name
    and the file path separated by ':',
    e.g. 'input1:file_path1,input2:file_path2'. In case the input layer name
    contains ':' e.g. "input:0", escape it with "::" literal,
    e.g. `input::0:file_path1`. If a input name appears in both
    `input_layer_value_range` and `input_layer_value_files`, the corresponding
    input value range specified by`input_layer_value_range` will be ignored.
    The file format is binary, and the content should be either a byte array or
    null-separated strings. Note that the input layer name must also exist in
    the list of names specified by `input_layer`.

### TFLite delegate parameters
The tool supports all runtime/delegate parameters introduced by
[the delegate registrar](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates).
The following simply lists the names of these parameters and additional notes
where applicable. For details about each parameter, please refer to
[this page](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar).
#### Common parameters
* `max_delegated_partitions`: `int` (default=0)
* `min_nodes_per_partition`:`int` (default=0)
* `delegate_serialize_dir`: `str` (default="")
* `delegate_serialize_token`: `str` (default="")

#### GPU delegate
* `use_gpu`: `bool` (default=false)
* `gpu_precision_loss_allowed`: `bool` (default=true)
* `gpu_experimental_enable_quant`: `bool` (default=true)
* `gpu_inference_for_sustained_speed`: `bool` (default=false)
* `gpu_backend`: `string` (default="")
* `gpu_wait_type`: `str` (default="")

#### NNAPI delegate

*   `use_nnapi`: `bool` (default=false) \
    Note some Android P devices will fail to use NNAPI for models in
    `/data/local/tmp/` and this benchmark tool will not correctly use NNAPI.
*   `nnapi_execution_preference`: `str` (default="") \
    Should be one of: `fast_single_answer`, `sustained_speed`, `low_power`,
    `undefined`.
*   `nnapi_execution_priority`: `str` (default="") \
    Note this requires Android 11+.
*   `nnapi_accelerator_name`: `str` (default="") \
    Note this requires Android 10+.
*   `disable_nnapi_cpu`: `bool` (default=true)
*   `nnapi_allow_fp16`: `bool` (default=false)
*   `nnapi_allow_dynamic_dimensions`:`bool` (default=false)
*   `nnapi_use_burst_mode`:`bool` (default=false)

#### Hexagon delegate
* `use_hexagon`: `bool` (default=false)
* `hexagon_profiling`: `bool` (default=false) \
Note enabling this option will not produce profiling results outputs unless
`enable_op_profiling` is also turned on. When both parameters are set to true,
the profile of ops on hexagon DSP will be added to the profile table. Note that,
the reported data on hexagon is in cycles, not in ms like on cpu.
* `hexagon_lib_path`: `string` (default="/data/local/tmp/") \
The library path for the underlying Hexagon libraries.
This is where libhexagon_nn_skel*.so files should be.
For libhexagon_interface.so it needs to be on a path that can be loaded from
example: put it in LD_LIBRARY_PATH.

#### XNNPACK delegate
*   `use_xnnpack`: `bool` (default=false) \
Note if this option is explicitly set to `false`, the TfLite runtime will use
its original CPU kernels for model execution. In other words, after enabling
the feature that the XNNPACK delegate is applied by default in TfLite runtime,
explictly setting this flag to `false` will cause the benchmark tool to disable
the feature at runtime, and to use the original non-delegated CPU execution path
for model benchmarking.

#### CoreML delegate
*   `use_coreml`: `bool` (default=false)
*   `coreml_version`: `int` (default=0)

#### External delegate
*   `external_delegate_path`: `string` (default="")
*   `external_delegate_options`: `string` (default="")

#### Stable delegate [Experimental]
*   `stable_delegate_loader_settings`: `string` (default="") A path to the
    JSON-encoded delegate [`TFLiteSettings`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/acceleration/configuration/configuration.proto#L488) file, which is defined in `configuration.proto`.

As some delegates are only available on certain platforms, when running the
benchmark tool on a particular platform, specifying `--help` will print out all
supported parameters.

### Use multiple delegates
When multiple delegates are specified to be used in the commandline flags, the
order of delegates applied to the TfLite runtime will be same as their enabling
commandline flag is specified. For example, "--use_xnnpack=true --use_gpu=true"
means applying the XNNPACK delegate first, and then the GPU delegate secondly.
In comparison, "--use_gpu=true --use_xnnpack=true" means applying the GPU
delegate first, and then the XNNPACK delegate secondly.

## To build/install/run

### On Android:

(0) Refer to https://www.tensorflow.org/lite/guide/build_android to edit the
`WORKSPACE` to configure the android NDK/SDK.

(1) Build for your specific platform, e.g.:

```
bazel build -c opt \
  --config=android_arm64 \
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
bazel build --config=android_arm64 \
  tensorflow/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so
adb push bazel-bin/tensorflow/lite/delegates/hexagon/hexagon_nn/libhexagon_interface.so /data/local/tmp
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

## Build the benchmark tool with Tensorflow ops support

You can build the benchmark tool with [Tensorflow operators support](https://www.tensorflow.org/lite/guide/ops_select).

### How to build

To build the tool, you need to use 'benchmark_model_plus_flex' target with
'--config=monolithic' option.

```
bazel build -c opt \
  --config=monolithic \
  tensorflow/lite/tools/benchmark:benchmark_model_plus_flex
```

### How to benchmark tflite model with Tensorflow ops

Tensorflow ops support just works the benchmark tool is built with Tensorflow
ops support. It doesn't require any additional option to use it.

```
bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex \
  --graph=model_converted_with_TF_ops.tflite \
```
