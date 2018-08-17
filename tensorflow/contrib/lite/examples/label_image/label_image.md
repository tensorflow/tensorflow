label_image for TensorFlow Lite inspired by TensorFlow's label_image.

To build label_image for Android, run $TENSORFLOW_ROOT/configure 
and set Android NDK or add them to `$TENSORFLOW_ROOT/.tf_configure.bazelrc`.
 
To build it for Android ARMv8:
```
> bazel build --config monolithic --cxxopt=-std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=arm64-v8a \
  //tensorflow/contrib/lite/examples/label_image:label_image
```
or
```
> bazel build --config android_arm64 --config monolithic --cxxopt=-std=c++11 \
  //tensorflow/contrib/lite/examples/label_image:label_image
```

To build it for Android arm-v7a:
```
> bazel build --config monolithic --cxxopt=-std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=armeabi-v7a \
  //tensorflow/contrib/lite/examples/label_image:label_image
```
or
```
> bazel build --config android_arm --config monolithic --cxxopt=-std=c++11 \
  //tensorflow/contrib/lite/examples/label_image:label_image
```

Build it for desktop machines (tested on Ubuntu and OS X)
```
> bazel build --config opt --cxxopt=-std=c++11 //tensorflow/contrib/lite/examples/label_image:label_image
```
To run it. Prepare `./mobilenet_quant_v1_224.tflite`, `./grace_hopper.bmp`, and `./labels.txt`.

```
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/contrib/lite/examples/label_image/testdata/grace_hopper.bmp > grace_hopper.bmp

curl  https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv mobilenet_v1_1.0_224/labels.txt
mv mobilenet_v1_1.0_224/labels.txt .

curl http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz | tar xzv ./mobilenet_v1_1.0_224_quant.tflite
mv mobilenet_v1_1.0_224_quant.tflite mobilenet_quant_v1_224.tflite
```


Run it:
```
> ./label_image
Loaded model ./mobilenet_quant_v1_224.tflite
resolved reporter
invoked
average time: 100.986 ms 
0.439216: 653 military uniform
0.372549: 458 bow tie
0.0705882: 466 bulletproof vest
0.0235294: 514 cornet
0.0196078: 835 suit
```
Run `interpreter->Invoker()` 100 times:
```
> ./label_image -c 100
Loaded model ./mobilenet_quant_v1_224.tflite
resolved reporter
invoked
average time: 33.4694 ms
...
```

Run a floating point (`mobilenet_v1_1.0_224.tflite`) model,
```
> ./label_image -m mobilenet_v1_1.0_224.tflite
Loaded model mobilenet_v1_1.0_224.tflite
resolved reporter
invoked
average time: 263.493 ms 
0.88615: 653 military uniform
0.0422316: 440 bearskin
0.0109948: 466 bulletproof vest
0.0105327: 401 academic gown
0.00947104: 723 ping-pong bal
```

To run profiling with TF Lite profiling mechanism, build with
`--cxxopt=-DTFLITE_PROFILING_ENABLED`
e.g.,

```
bazel build --config opt \
--cxxopt=-std=c++11 \
--cxxopt=-DTFLITE_PROFILING_ENABLED \
--config monolithic \
//tensorflow/contrib/lite/examples/label_image:label_image

```

Run it with `-p`

```
> ./label_image  -p 1
Loaded model ./mobilenet_quant_v1_224.tflite
resolved reporter
invoked
average time: 55.522 ms
     4.702, Node   0, OpCode   3, CONV_2D
     9.158, Node   1, OpCode   4, DEPTHWISE_CONV_2D
     3.295, Node   2, OpCode   3, CONV_2D
     3.431, Node   3, OpCode   4, DEPTHWISE_CONV_2D
     1.708, Node   4, OpCode   3, CONV_2D
     5.836, Node   5, OpCode   4, DEPTHWISE_CONV_2D
     2.263, Node   6, OpCode   3, CONV_2D
     1.408, Node   7, OpCode   4, DEPTHWISE_CONV_2D
     1.105, Node   8, OpCode   3, CONV_2D
     2.678, Node   9, OpCode   4, DEPTHWISE_CONV_2D
     1.784, Node  10, OpCode   3, CONV_2D
     0.665, Node  11, OpCode   4, DEPTHWISE_CONV_2D
     0.836, Node  12, OpCode   3, CONV_2D
     1.198, Node  13, OpCode   4, DEPTHWISE_CONV_2D
     1.494, Node  14, OpCode   3, CONV_2D
     1.208, Node  15, OpCode   4, DEPTHWISE_CONV_2D
     1.429, Node  16, OpCode   3, CONV_2D
     1.202, Node  17, OpCode   4, DEPTHWISE_CONV_2D
     1.477, Node  18, OpCode   3, CONV_2D
     1.193, Node  19, OpCode   4, DEPTHWISE_CONV_2D
     1.415, Node  20, OpCode   3, CONV_2D
     1.209, Node  21, OpCode   4, DEPTHWISE_CONV_2D
     1.394, Node  22, OpCode   3, CONV_2D
     0.288, Node  23, OpCode   4, DEPTHWISE_CONV_2D
     0.770, Node  24, OpCode   3, CONV_2D
     0.542, Node  25, OpCode   4, DEPTHWISE_CONV_2D
     1.453, Node  26, OpCode   3, CONV_2D
     0.020, Node  27, OpCode   1, AVERAGE_POOL_2D
     0.303, Node  28, OpCode   3, CONV_2D
     0.000, Node  29, OpCode  43, SQUEEZE
     0.052, Node  30, OpCode  25, SOFTMAX
0.365: 907 907:Windsor tie
0.365: 653 653:military uniform
0.043: 668 668:mortarboard
0.035: 458 458:bow tie, bow-tie, bowtie
0.027: 543 543:drumstick
```

See the source code for other command line options.
