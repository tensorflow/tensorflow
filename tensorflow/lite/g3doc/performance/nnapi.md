# TensorFlow Lite NNAPI delegate

The
[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks)
is available on all Android devices running Android 8.1 (API level 27) or
higher. It provides acceleration for TensorFlow Lite models on Android devices
with supported hardware accelerators including:

*   Graphics Processing Unit (GPU)
*   Digital Signal Processor (DSP)
*   Neural Processing Unit (NPU)

Performance will vary depending on the specific hardware available on device.

This page describes how to use the NNAPI delegate with the TensorFlow Lite
Interpreter in Java and Kotlin. For Android C APIs, please refer to
[Android Native Developer Kit documentation](https://developer.android.com/ndk/guides/neuralnetworks).

## Trying the NNAPI delegate on your own model

### Gradle import

The NNAPI delegate is part of the TensorFlow Lite Android interpreter, release
1.14.0 or higher. You can import it to your project by adding the following to
your module gradle file:

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### Initializing the NNAPI delegate

Add the code to initialize the NNAPI delegate before you initialize the
TensorFlow Lite interpreter.

Note: Although NNAPI is supported from API Level 27 (Android Oreo MR1), the
support for operations improved significantly for API Level 28 (Android Pie)
onwards. As a result, we recommend developers use the NNAPI delegate for Android
Pie or above for most scenarios.

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

Interpreter.Options options = (new Interpreter.Options());
NnApiDelegate nnApiDelegate = null;
// Initialize interpreter with NNAPI delegate for Android Pie or above
if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);
}

// Initialize TFLite interpreter
try {
    tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
} catch (Exception e) {
    throw new RuntimeException(e);
}

// Run inference
// ...

// Unload delegate
tfLite.close();
if(null != nnApiDelegate) {
    nnApiDelegate.close();
}
```

## Best practices

### Test performance before deploying

Runtime performance can vary significantly due to model architecture, size,
operations, hardware availability, and runtime hardware utilization. For
example, if an app heavily utilizes the GPU for rendering, NNAPI acceleration
may not improve performance due to resource contention. We recommend running a
simple performance test using the debug logger to measure inference time. Run
the test on several phones with different chipsets (manufacturer or models from
the same manufacturer) that are representative of your user base before enabling
NNAPI in production.

For advanced developers, TensorFlow Lite also offers
[a model benchmark tool for Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

### Create a device exclusion list

In production, there may be cases where NNAPI does not perform as expected. We
recommend developers maintain a list of devices that should not use NNAPI
acceleration in combination with particular models. You can create this list
based on the value of `"ro.board.platform"`, which you can retrieve using the
following code snippet:

```java
String boardPlatform = "";

try {
    Process sysProcess =
        new ProcessBuilder("/system/bin/getprop", "ro.board.platform").
        redirectErrorStream(true).start();

    BufferedReader reader = new BufferedReader
        (new InputStreamReader(sysProcess.getInputStream()));
    String currentLine = null;

    while ((currentLine=reader.readLine()) != null){
        boardPlatform = line;
    }
    sysProcess.destroy();
} catch (IOException e) {}

Log.d("Board Platform", boardPlatform);
```

For advanced developers, consider maintaining this list via a remote
configuration system. The TensorFlow team is actively working on ways to
simplify and automate discovering and applying the optimal NNAPI configuration.

### Quantization

Quantization reduces model size by using 8-bit integers or 16-bit floats instead
of 32-bit floats for computation. 8-bit integer model sizes are a quarter of the
32-bit float versions; 16-bit floats are half of the size. Quantization can
improve performance significantly though the process could trade off some model
accuracy.

There are multiple types of post-training quantization techniques available,
but, for maximum support and acceleration on current hardware, we recommend
[full integer quantization](post_training_quantization#full_integer_quantization_of_weights_and_activations).
This approach converts both the weight and the operations into integers. This
quantization process requires a representative dataset to work.

### Use supported models and ops

If the NNAPI delegate does not support some of the ops or parameter combinations
in a model, the framework only runs the supported parts of the graph on the
accelerator. The remainder runs on the CPU, which results in split execution.
Due to the high cost of CPU/accelerator synchronization, this may result in
slower performance than executing the whole network on the CPU alone.

NNAPI performs best when models only use
[supported ops](https://developer.android.com/ndk/guides/neuralnetworks#model).
The following models are known to be compatible with NNAPI:

*   [MobileNet v1 (224x224) image classification (float model download)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html)
    [(quantized model download)](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)
    \
    _(image classification model designed for mobile and embedded based vision
    applications)_
*   [MobileNet v2 SSD object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html)
    [(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)
    \
    _(image classification model that detects multiple objects with bounding
    boxes)_
*   [MobileNet v1(300x300) Single Shot Detector (SSD) object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html)
[(download)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
*   [PoseNet for pose estimation](https://github.com/tensorflow/tfjs-models/tree/master/posenet)
    [(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)
    \
    _(vision model that estimates the poses of a person(s) in image or video)_

NNAPI acceleration is also not supported when the model contains
dynamically-sized outputs. In this case, you will get a warning like:

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors.
```
