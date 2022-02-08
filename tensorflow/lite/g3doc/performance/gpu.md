# TensorFlow Lite GPU delegate

[TensorFlow Lite](https://www.tensorflow.org/lite) supports several hardware
accelerators. This document describes how to use the GPU backend using the
TensorFlow Lite delegate APIs on Android and iOS.

GPUs are designed to have high throughput for massively parallelizable
workloads. Thus, they are well-suited for deep neural nets, which consist of a
huge number of operators, each working on some input tensor(s) that can be
easily divided into smaller workloads and carried out in parallel, typically
resulting in lower latency. In the best scenario, inference on the GPU may now
run fast enough for previously not available real-time applications.

Unlike CPUs, GPUs compute with 16-bit or 32-bit floating point numbers and do
not require quantization for optimal performance. The delegate does accept 8-bit
quantized models, but the calculation will be performed in floating point
numbers. Refer to the [advanced documentation](gpu_advanced.md) for details.

Another benefit with GPU inference is its power efficiency. GPUs carry out the
computations in a very efficient and optimized manner, so that they consume less
power and generate less heat than when the same task is run on CPUs.

## Demo app tutorials

The easiest way to try out the GPU delegate is to follow the below tutorials,
which go through building our classification demo applications with GPU support.
The GPU code is only binary for now; it will be open-sourced soon. Once you
understand how to get our demos working, you can try this out on your own custom
models.

### Android (with Android Studio)

For a step-by-step tutorial, watch the
[GPU Delegate for Android](https://youtu.be/Xkhgre8r5G0) video.

Note: This requires OpenCL or OpenGL ES (3.1 or higher).

#### Step 1. Clone the TensorFlow source code and open it in Android Studio

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### Step 2. Edit `app/build.gradle` to use the nightly GPU AAR

Note: You can now target **Android S+** with `targetSdkVersion="S"` in your
manifest, or `targetSdkVersion "S"` in your Gradle `defaultConfig` (API level
TBD). In this case, you should merge the contents of
[`AndroidManifestGpu.xml`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/AndroidManifestGpu.xml)
into your Android application's manifest. Without this change, the GPU delegate
cannot access OpenCL libraries for acceleration. *AGP 4.2.0 or above is required
for this to work.*

Add the `tensorflow-lite-gpu` package alongside the existing `tensorflow-lite`
package in the existing `dependencies` block.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### Step 3. Build and run

Run → Run ‘app’. When you run the application you will see a button for enabling
the GPU. Change from quantized to a float model and then click GPU to run on the
GPU.

![running android gpu demo and switch to gpu](images/android_gpu_demo.gif)

### iOS (with XCode)

For a step-by-step tutorial, watch the
[GPU Delegate for iOS](https://youtu.be/a5H4Zwjp49c) video.

Note: This requires XCode v10.1 or later.

#### Step 1. Get the demo source code and make sure it compiles.

Follow our iOS Demo App [tutorial](https://www.tensorflow.org/lite/guide/ios).
This will get you to a point where the unmodified iOS camera demo is working on
your phone.

#### Step 2. Modify the Podfile to use the TensorFlow Lite GPU CocoaPod

From 2.3.0 release, by default GPU delegate is excluded from the pod to reduce
the binary size. You can include them by specifying subspec. For
`TensorFlowLiteSwift` pod:

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

OR

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

You can do similarly for `TensorFlowLiteObjC` or `TensorFlowLitC` if you want to
use the Objective-C (from 2.4.0 release) or C API.

<div>
  <devsite-expandable>
    <h4 class="showalways">Before 2.3.0 release</h4>
    <h4>Until TensorFlow Lite 2.0.0</h4>
    <p>
      We have built a binary CocoaPod that includes the GPU delegate. To switch
      the project to use it, modify the
      `tensorflow/tensorflow/lite/examples/ios/camera/Podfile` file to use the
      `TensorFlowLiteGpuExperimental` pod instead of `TensorFlowLite`.
    </p>
    <pre class="prettyprint lang-ruby notranslate" translate="no"><code>
    target 'YourProjectName'
      # pod 'TensorFlowLite', '1.12.0'
      pod 'TensorFlowLiteGpuExperimental'
    </code></pre>
    <h4>Until TensorFlow Lite 2.2.0</h4>
    <p>
      From TensorFlow Lite 2.1.0 to 2.2.0, GPU delegate is included in the
      `TensorFlowLiteC` pod. You can choose between `TensorFlowLiteC` and
      `TensorFlowLiteSwift` depending on the language.
    </p>
  </devsite-expandable>
</div>

#### Step 3. Enable the GPU delegate

To enable the code that will use the GPU delegate, you will need to change
`TFLITE_USE_GPU_DELEGATE` from 0 to 1 in `CameraExampleViewController.h`.

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### Step 4. Build and run the demo app

After following the previous step, you should be able to run the app.

#### Step 5. Release mode

While in Step 4 you ran in debug mode, to get better performance, you should
change to a release build with the appropriate optimal Metal settings. In
particular, To edit these settings go to the `Product > Scheme > Edit
Scheme...`. Select `Run`. On the `Info` tab, change `Build Configuration`, from
`Debug` to `Release`, uncheck `Debug executable`.

![setting up release](images/iosdebug.png)

Then click the `Options` tab and change `GPU Frame Capture` to `Disabled` and
`Metal API Validation` to `Disabled`.

![setting up metal options](images/iosmetal.png)

Lastly make sure to select Release-only builds on 64-bit architecture. Under
`Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example
-> Build Settings` set `Build Active Architecture Only > Release` to Yes.

![setting up release options](images/iosrelease.png)

## Trying the GPU delegate on your own model

### Android

Note: The TensorFlow Lite Interpreter must be created on the same thread as
where it is run. Otherwise, `TfLiteGpuDelegate Invoke: GpuDelegate must run on
the same thread where it was initialized.` may occur.

There are two ways to invoke model acceleration depending on if you are using
[Android Studio ML Model Binding](../inference_with_metadata/codegen#acceleration)
or TensorFlow Lite Interpreter.

#### TensorFlow Lite Interpreter

Look at the demo to see how to add the delegate. In your application, add the
AAR as above, import `org.tensorflow.lite.gpu.GpuDelegate` module, and use
the`addDelegate` function to register the GPU delegate to the interpreter:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p><pre class="prettyprint lang-kotlin">
    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
        if(compatList.isDelegateSupportedOnThisDevice){
            // if the device has a supported GPU, add the GPU delegate
            val delegateOptions = compatList.bestOptionsForThisDevice
            this.addDelegate(GpuDelegate(delegateOptions))
        } else {
            // if the GPU is not supported, run on 4 threads
            this.setNumThreads(4)
        }
    }

    val interpreter = Interpreter(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
    import org.tensorflow.lite.Interpreter;
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
    } else {
        // if the GPU is not supported, run on 4 threads
        options.setNumThreads(4);
    }

    Interpreter interpreter = new Interpreter(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre></p>
    </section>
  </devsite-selector>
</div>

### iOS

Note: GPU delegate can also use C API for Objective-C code. Prior to TensorFlow
Lite 2.4.0 release, this was the only option.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p><pre class="prettyprint lang-swift">
    import TensorFlowLite

    // Load model ...

    // Initialize TensorFlow Lite interpreter with the GPU delegate.
    let delegate = MetalDelegate()
    if let interpreter = try Interpreter(modelPath: modelPath,
                                         delegates: [delegate]) {
      // Run inference ...
    }
      </pre></p>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p><pre class="prettyprint lang-objc">
    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    #import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"
    #import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize GPU delegate
    TFLMetalDelegate* metalDelegate = [[TFLMetalDelegate alloc] init];

    // Initialize interpreter with model path and GPU delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ metalDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
    ```
      </pre></p>
    </section>
    <section>
      <h3>C (Until 2.3.0)</h3>
      <p><pre class="prettyprint lang-c">
    #include "tensorflow/lite/c/c_api.h"
    #include "tensorflow/lite/delegates/gpu/metal_delegate.h"

    // Initialize model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with GPU delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TFLGPUDelegateCreate(nil);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, metal_delegate);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreterAllocateTensors(interpreter);

    NSMutableData *input_data = [NSMutableData dataWithLength:input_size * sizeof(float)];
    NSMutableData *output_data = [NSMutableData dataWithLength:output_size * sizeof(float)];
    TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // Run inference
    TfLiteTensorCopyFromBuffer(input, inputData.bytes, inputData.length);
    TfLiteInterpreterInvoke(interpreter);
    TfLiteTensorCopyToBuffer(output, outputData.mutableBytes, outputData.length);

    // Clean up
    TfLiteInterpreterDelete(interpreter);
    TFLGpuDelegateDelete(metal_delegate);
    TfLiteModelDelete(model);
      </pre></p>
    </section>
  </devsite-selector>
</div>

## Supported Models and Ops

With the release of the GPU delegate, we included a handful of models that can
be run on the backend:

*   [MobileNet v1 (224x224) image classification](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite)
    <br /><i>(image classification model designed for mobile and embedded based vision applications)</i>
*   [DeepLab segmentation (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)
    <br /><i>(image segmentation model that assigns semantic labels (e.g., dog, cat, car) to every pixel in the input image)</i>
*   [MobileNet SSD object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)
    <br /><i>(image classification model that detects multiple objects with bounding boxes)</i>
*   [PoseNet for pose estimation](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)
    <br /><i>(vision model that estimates the poses of a person(s) in image or video)</i>

To see a full list of supported ops, please see the
[advanced documentation](gpu_advanced.md).

## Non-supported models and ops

If some of the ops are not supported by the GPU delegate, the framework will
only run a part of the graph on the GPU and the remaining part on the CPU. Due
to the high cost of CPU/GPU synchronization, a split execution mode like this
will often result in slower performance than when the whole network is run on
the CPU alone. In this case, the user will get a warning like:

```none
WARNING: op code #42 cannot be handled by this delegate.
```

We did not provide a callback for this failure, as this is not a true run-time
failure, but something that the developer can observe while trying to get the
network to run on the delegate.

## Tips for optimization

### Optimizing for mobile devices

Some operations that are trivial on the CPU may have a high cost for the GPU on
mobile devices. Reshape operations are particularly expensive to run, including
`BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH`, and so forth. You should
closely examine use of reshape operations, and consider that may have been
applied only for exploring data or for early iterations of your model. Removing
them can significantly improve performance.

On GPU, tensor data is sliced into 4-channels. Thus, a computation on a tensor
of shape `[B,H,W,5]` will perform about the same on a tensor of shape
`[B,H,W,8]` but significantly worse than `[B,H,W,4]`. In that sense, if the
camera hardware supports image frames in RGBA, feeding that 4-channel input is
significantly faster as a memory copy (from 3-channel RGB to 4-channel RGBX) can
be avoided.

For best performance, you should consider retraining the classifier with a
mobile-optimized network architecture. Optimization for on-device inferencing
can dramatically reduce latency and power consumption by taking advantage of
mobile hardware features.

### Reducing initialization time with serialization

The GPU delegate feature allows you to load from pre-compiled kernel code and
model data serialized and saved on disk from previous runs. This approach avoids
re-compilation and reduces startup time by up to 90%. For instructions on how to
apply serialization to your project, see
[GPU Delegate Serialization](gpu_advanced.md#gpu_delegate_serialization).
