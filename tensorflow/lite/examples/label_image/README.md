# TensorFlow Lite C++ image classification demo

This example shows how you can load a pre-trained and converted
TensorFlow Lite model and use it to recognize objects in images.

Before you begin,
make sure you [have TensorFlow installed](https://www.tensorflow.org/install).

You also need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) in order
to build this example code. And be sure you have the Python `future` module
installed:

```
pip install future --user
```

## Build the example

First run `$TENSORFLOW_ROOT/configure`. To build for Android, set
Android NDK or configure NDK setting in
`$TENSORFLOW_ROOT/WORKSPACE` first.

Build it for desktop machines (tested on Ubuntu and OS X):

```
bazel build -c opt //tensorflow/lite/examples/label_image:label_image
```

Build it for Android ARMv8:

```
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/examples/label_image:label_image
```

Build it for Android arm-v7a:

```
bazel build -c opt --config=android_arm \
  //tensorflow/lite/examples/label_image:label_image
```

## Download sample model and image

You can use any compatible model, but the following MobileNet v1 model offers
a good demonstration of a model trained to recognize 1,000 different objects.

```
# Get model
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp

# Get labels
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt

mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```

## Run the sample on a desktop

```
bazel-bin/tensorflow/lite/examples/label_image/label_image \
  --tflite_model /tmp/mobilenet_v1_1.0_224.tflite \
  --labels /tmp/labels.txt \
  --image tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp
```

You should see results like this:

```
Loaded model /tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
invoked
average time: 68.12 ms
0.860174: 653 653:military uniform
0.0481017: 907 907:Windsor tie
0.00786704: 466 466:bulletproof vest
0.00644932: 514 514:cornet, horn, trumpet, trump
0.00608029: 543 543:drumstick
```

## Run the sample on an Android device

Prepare data on devices, e.g.,

```
adb push bazel-bin/tensorflow/lite/examples/label_image/label_image  /data/local/tmp
adb push /tmp/mobilenet_v1_1.0_224.tflite  /data/local/tmp
adb push tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp  /data/local/tmp
adb push /tmp/labels.txt /data/local/tmp
```

Run it,

```
adb shell "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt"
```

then you should see something like the following:

```
Loaded model /data/local/tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime.
invoked
average time: 25.03 ms
0.907071: 653 military uniform
0.0372416: 907 Windsor tie
0.00733753: 466 bulletproof vest
0.00592852: 458 bow tie
0.00414091: 514 cornet
```

Run the model with NNAPI delegate (`-a 1`),

```
adb shell "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -a 1 -f 1"
```

then you should see something like the following:

```
Loaded model /data/local/tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for NNAPI. Applied NNAPI delegate.
invoked
average time:10.348 ms
0.905401: 653 military uniform
0.0379589: 907 Windsor tie
0.00735866: 466 bulletproof vest
0.00605307: 458 bow tie
0.00422573: 514 cornet
```

To run a model with the Hexagon Delegate, assuming we have followed the
[Hexagon Delegate Guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/hexagon_delegate.md)
and installed Hexagon libraries in `/data/local/tmp`. Run it with (`-j 1`)

```
adb shell \
    "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -j 1"
```

then you should see something like the followings:

```
Loaded model /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime.
loaded libcdsprpc.so
INFO: Created TensorFlow Lite delegate for Hexagon.
INFO: Hexagon delegate: 31 nodes delegated out of 31 nodes with 1 partitions.
Applied Hexagon delegate.
invoked
average time: 4.231 ms
0.639216: 458 bow tie
0.329412: 653 military uniform
0.00784314: 835 suit
0.00784314: 611 jersey
0.00392157: 514 cornet
```

Run the model with the XNNPACK delegate (`-x 1`),

```shell
adb shell \
    "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -x 1"
```

then you should see something like the following:

```
Loaded model /data/local/tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime. Applied XNNPACK delegate.
invoked
average time: 17.33 ms
0.90707: 653 military uniform
0.0372418: 907 Windsor tie
0.0073376: 466 bulletproof vest
0.00592857: 458 bow tie
0.00414093: 514 cornet
```

With `-h` or any other unsupported flags, `label_image` will list supported
options:

```shell
sargo:/data/local/tmp $ ./label_image -h
./label_image: invalid option -- h
label_image
--accelerated, -a: [0|1], use Android NNAPI or not
--old_accelerated, -d: [0|1], use old Android NNAPI delegate or not
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--count, -c: loop interpreter->Invoke() for certain times
--gl_backend, -g: [0|1]: use GPU Delegate on Android
--hexagon_delegate, -j: [0|1]: use Hexagon Delegate on Android
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--image, -i: image_name.bmp
--labels, -l: labels for the model
--tflite_model, -m: model_name.tflite
--profiling, -p: [0|1], profiling or not
--num_results, -r: number of results to show
--threads, -t: number of threads
--verbose, -v: [0|1] print more information
--warmup_runs, -w: number of warmup runs
--xnnpack_delegate, -x [0:1]: xnnpack delegate`
```

See the `label_image.cc` source code for other command line options.

Note that this binary also supports more runtime/delegate arguments introduced
by the [delegate registrar](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates).
If there is any conflict, the arguments mentioned earlier are given precedence.
For example, you can run the binary with additional command line options
such as `--use_nnapi=true --nnapi_accelerator_name=google-edgetpu` to utilize
the EdgeTPU in a 4th-gen Pixel phone. Please be aware that the "=" in the option
should not be omitted.

```
adb shell \
    "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -j 1 \
    --use_nnapi=true --nnapi_accelerator_name=google-edgetpu"
```
