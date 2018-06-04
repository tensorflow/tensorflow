# TFLite Model Benchmark Tool

## Description

A simple C++ binary to benchmark a TFLite model and its individual operators,
both on desktop machines and on Android.

## To build/install/run

### On Android:

(0) Refer to https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android to edit the `WORKSPACE` to configure the android NDK/SDK.

(1) Build for your specific platform, e.g.:

```
bazel build -c opt \
  --config=android_arm \
  --cxxopt='--std=c++11' \
  tensorflow/contrib/lite/tools/benchmark:benchmark_model
```

(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):

```
adb push bazel-bin/tensorflow/contrib/lite/tools/benchmark/benchmark_model /data/local/tmp
```

(3) Make the binary executable.

```
adb shell chmod +x /data/local/tmp/benchmark_model
```

(4) Push the compute graph that you need to test. For example:

```
adb push mobilenet_quant_v1_224.tflite /data/local/tmp
```

(5) Run the benchmark. For example:

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --input_layer="Placeholder" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="uint8" \
  --output_layer="MobilenetV1/Predictions/Reshape_1" \
  --num_threads=4
```

### On desktop:
(1) build the binary

```
bazel build -c opt tensorflow/contrib/lite/tools/benchmark:benchmark_model
```

(2) Run on your compute graph, similar to the Android case but without the need of adb shell.
For example:

```
bazel-bin/tensorflow/contrib/lite/tools/benchmark/benchmark_model \
  --graph=mobilenet_quant_v1_224.tflite \
  --input_layer="Placeholder" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="uint8" \
  --output_layer="MobilenetV1/Predictions/Reshape_1" \
  --num_threads=4
```

The MobileNet graph used as an example here may be downloaded from
https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip

## Profiling model operators
The benchmark model binary also allows you to profile operators and give execution times of each operator. To do this,
compile the binary with a compiler flag that enables profiling to be compiled in. Pass **--copt=-DTFLITE_PROFILING_ENABLED**
to compile benchmark with profiling support.
For example, to compile with profiling support on Android, add this flag to the previous command:

```
bazel build -c opt \
  --config=android_arm \
  --cxxopt='--std=c++11' \
  --copt=-DTFLITE_PROFILING_ENABLED \
  tensorflow/contrib/lite/tools/benchmark:benchmark_model
```
This compiles TFLite with profiling enabled, now you can run the benchmark binary like before. The binary will produce detailed statistics for each operation similar to those shown below:

```

============================== Run Order ==============================
	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 CONV_2D	    0.000	    9.132	    9.132	  0.121%	  0.121%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_0/Relu6]
	       DEPTHWISE_CONV_2D	    9.135	    3.280	    3.280	  0.043%	  0.165%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6]
	                 CONV_2D	   12.419	    6.877	    6.877	  0.091%	  0.256%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   19.299	    1.708	    1.708	  0.023%	  0.278%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6]
	                 CONV_2D	   21.012	    4.162	    4.162	  0.055%	  0.334%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   25.177	    3.520	    3.520	  0.047%	  0.380%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6]
	                 CONV_2D	   28.701	   10.218	   10.218	  0.136%	  0.516%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   38.922	    0.827	    0.827	  0.011%	  0.527%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6]
	                 CONV_2D	   39.752	    1.401	    1.401	  0.019%	  0.545%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   41.156	    1.290	    1.290	  0.017%	  0.563%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6]
	                 CONV_2D	   42.448	    5.995	    5.995	  0.080%	  0.642%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   48.445	    0.409	    0.409	  0.005%	  0.647%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6]
	                 CONV_2D	   48.856	    6.167	    6.167	  0.082%	  0.729%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   55.026	    0.629	    0.629	  0.008%	  0.738%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6]
	                 CONV_2D	   55.656	    6.464	    6.464	  0.086%	  0.823%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   62.124	    0.647	    0.647	  0.009%	  0.832%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6]
	                 CONV_2D	   62.774	   14.666	   14.666	  0.195%	  1.026%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   77.444	    0.635	    0.635	  0.008%	  1.035%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6]
	                 CONV_2D	   78.081	    7.186	    7.186	  0.095%	  1.130%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   85.270	    0.646	    0.646	  0.009%	  1.139%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6]
	                 CONV_2D	   85.918	    9.529	    9.529	  0.126%	  1.265%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   95.451	    0.628	    0.628	  0.008%	  1.273%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6]
	                 CONV_2D	   96.081	    2.077	    2.077	  0.028%	  1.301%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   98.162	    0.168	    0.168	  0.002%	  1.303%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6]
	                 CONV_2D	   98.332	    1.007	    1.007	  0.013%	  1.317%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   99.342	    0.288	    0.288	  0.004%	  1.320%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6]
	                 CONV_2D	   99.632	    8.197	    8.197	  0.109%	  1.429%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6]
	         AVERAGE_POOL_2D	  107.832	    0.045	    0.045	  0.001%	  1.430%	     0.000	        0	[MobilenetV1/Logits/AvgPool_1a/AvgPool]
	                 CONV_2D	  107.878	    0.325	    0.325	  0.004%	  1.434%	     0.000	        0	[MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd]
	                 RESHAPE	  108.206	    0.003	    0.003	  0.000%	  1.434%	     0.000	        0	[MobilenetV1/Predictions/Reshape]
	                 SOFTMAX	  108.211	    0.038	    0.038	  0.001%	  1.434%	     0.000	        0	[MobilenetV1/Predictions/Softmax]

============================== Top by Computation Time ==============================
	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 CONV_2D	   62.774	   14.666	   14.666	  0.195%	  0.195%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
	                 CONV_2D	   28.701	   10.218	   10.218	  0.136%	  0.330%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6]
	                 CONV_2D	   85.918	    9.529	    9.529	  0.126%	  0.456%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6]
	                 CONV_2D	    0.000	    9.132	    9.132	  0.121%	  0.578%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_0/Relu6]
	                 CONV_2D	   99.632	    8.197	    8.197	  0.109%	  0.686%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6]
	                 CONV_2D	   78.081	    7.186	    7.186	  0.095%	  0.782%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
	                 CONV_2D	   12.419	    6.877	    6.877	  0.091%	  0.873%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6]
	                 CONV_2D	   55.656	    6.464	    6.464	  0.086%	  0.958%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
	                 CONV_2D	   48.856	    6.167	    6.167	  0.082%	  1.040%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6]
	                 CONV_2D	   42.448	    5.995	    5.995	  0.080%	  1.120%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6]

============================== Top by Memory Use ==============================
	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 SOFTMAX	  108.211	    0.038	    0.038	  0.001%	  0.001%	     0.000	        0	[MobilenetV1/Predictions/Softmax]
	                 RESHAPE	  108.206	    0.003	    0.003	  0.000%	  0.001%	     0.000	        0	[MobilenetV1/Predictions/Reshape]
	                 CONV_2D	   78.081	    7.186	    7.186	  0.095%	  0.096%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   77.444	    0.635	    0.635	  0.008%	  0.104%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6]
	                 CONV_2D	   62.774	   14.666	   14.666	  0.195%	  0.299%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   62.124	    0.647	    0.647	  0.009%	  0.307%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6]
	                 CONV_2D	   55.656	    6.464	    6.464	  0.086%	  0.393%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   55.026	    0.629	    0.629	  0.008%	  0.401%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6]
	                 CONV_2D	   48.856	    6.167	    6.167	  0.082%	  0.483%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6]
	       DEPTHWISE_CONV_2D	   48.445	    0.409	    0.409	  0.005%	  0.489%	     0.000	        0	[MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6]

Number of nodes executed: 31
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	                 CONV_2D	       15	     1.861	    86.679%	    86.679%	     0.000	        0
	       DEPTHWISE_CONV_2D	       13	     0.286	    13.321%	   100.000%	     0.000	        0
	                 SOFTMAX	        1	     0.000	     0.000%	   100.000%	     0.000	        0
	                 RESHAPE	        1	     0.000	     0.000%	   100.000%	     0.000	        0
	         AVERAGE_POOL_2D	        1	     0.000	     0.000%	   100.000%	     0.000	        0

Timings (microseconds): count=50 first=108164 curr=128308 min=102850 max=197072 avg=150805 std=24368
Memory (bytes): count=0
31 nodes observed


Average inference timings in us: Warmup: 135310, Init: 12123, no stats: 150988

```


