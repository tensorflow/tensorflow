label_image for TensorFlow Lite inspired by TensorFlow's label_image.

To build label_image for Android, run $TENSORFLOW_ROOT/configure 
and set Android NDK or configure NDK setting in 
$TENSORFLOW_ROOT/WORKSPACE first.
 
To build it for android ARMv8:
```
> bazel build --config monolithic --cxxopt=-std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=arm64-v8a \
  //tensorflow/lite/examples/label_image:label_image
```
or
```
> bazel build --config android_arm64 --config monolithic --cxxopt=-std=c++11 \
  //tensorflow/lite/examples/label_image:label_image
```

To build it for android arm-v7a:
```
> bazel build --config monolithic --cxxopt=-std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=armeabi-v7a \
  //tensorflow/lite/examples/label_image:label_image
```
or
```
> bazel build --config android_arm --config monolithic --cxxopt=-std=c++11 \
  //tensorflow/lite/examples/label_image:label_image
```

Build it for desktop machines (tested on Ubuntu and OS X)
```
> bazel build --config opt --cxxopt=-std=c++11 //tensorflow/lite/examples/label_image:label_image
```
To run it. Prepare `./mobilenet_quant_v1_224.tflite`, `./grace_hopper.bmp`, and `./labels.txt`.

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
Run `interpreter->Invoke()` 100 times:
```
> ./label_image   -c 100                               
Loaded model ./mobilenet_quant_v1_224.tflite
resolved reporter
invoked
average time: 33.4694 ms
...
```

Run a floating point (`mobilenet_v1_1.0_224.tflite`) model,
```
> ./label_image -f 1 -m mobilenet_v1_1.0_224.tflite
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

See the source code for other command line options.
