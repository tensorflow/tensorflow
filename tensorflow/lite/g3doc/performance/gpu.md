# GPU delegates for TensorFlow Lite

Using graphics processing units (GPUs) to run your machine learning (ML) models
can dramatically improve the performance of your model and the user experience
of your ML-enabled applications. TensorFlow Lite enables the use of GPUs and
other specialized processors through hardware driver called
[*delegates*](./delegates). Enabling use of GPUs with your TensorFlow Lite ML
applications can provide the following benefits:

*   **Speed** - GPUs are built for high throughput of massively parallel
    workloads. This design makes them well-suited for deep neural nets, which
    consist of a huge number of operators, each working on input tensors that
    can be processed in parallel, which typically results in lower latency. In
    the best scenario, running your model on a GPU may run fast enough to enable
    real-time applications that were not previously possible.
*   **Power efficiency** - GPUs carry out ML computations in a very efficient
    and optimized manner, typically consuming less power and generating less
    heat than the same task running on CPUs.

This document provides an overview of GPUs support in TensorFlow Lite, and some
advanced uses for GPU processors. For more specific information about
implementing GPU support on specific platforms, see the following guides:

*   [GPU support for Android](../android/delegates/gpu)
*   [GPU support for iOS](../ios/delegates/gpu)

## GPU ML operations support {:#supported_ops}

There are some limitations to what TensorFlow ML operations, or *ops*, can be
accelerated by the TensorFlow Lite GPU delegate. The delegate supports the
following ops in 16-bit and 32-bit float precision:

*   `ADD`
*   `AVERAGE_POOL_2D`
*   `CONCATENATION`
*   `CONV_2D`
*   `DEPTHWISE_CONV_2D v1-2`
*   `EXP`
*   `FULLY_CONNECTED`
*   `LOGICAL_AND`
*   `LOGISTIC`
*   `LSTM v2 (Basic LSTM only)`
*   `MAX_POOL_2D`
*   `MAXIMUM`
*   `MINIMUM`
*   `MUL`
*   `PAD`
*   `PRELU`
*   `RELU`
*   `RELU6`
*   `RESHAPE`
*   `RESIZE_BILINEAR v1-3`
*   `SOFTMAX`
*   `STRIDED_SLICE`
*   `SUB`
*   `TRANSPOSE_CONV`

By default, all ops are only supported at version 1. Enabling the
[quantization support](#quantized-models) enables the appropriate versions, for
example, ADD v2.

### Troubleshooting GPU support

If some of the ops are not supported by the GPU delegate, the framework will
only run a part of the graph on the GPU and the remaining part on the CPU. Due
to the high cost of CPU/GPU synchronization, a split execution mode like this
often results in slower performance than when the whole network is run on
the CPU alone. In this case, the application generates warning, such as:

```none
WARNING: op code #42 cannot be handled by this delegate.
```

There is no callback for failures of this type, since this is not an actual
run-time failure. When testing execution of your model with the GPU delegate,
you should be alert for these warnings. A high number of these warnings can
indicate that your model is not the best fit for use for GPU acceleration, and
may require refactoring of the model.

## Example models

The following example models are built to take advantage GPU acceleration with
TensorFlow Lite and are provided for reference and testing:

*   [MobileNet v1 (224x224) image classification](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) -
    An image classification model designed for mobile and embedded based vision
    applications.
    ([model](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5))
*   [DeepLab segmentation (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) -
    image segmentation model that assigns semantic labels, such as a dog, cat,
    car, to every pixel in the input image.
    ([model](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1))
*   [MobileNet SSD object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) -
    An image classification model that detects multiple objects with bounding
    boxes.
    ([model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite))
*   [PoseNet for pose estimation](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) -
    A vision model that estimates the poses of people in image or video.
    ([model](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1))

## Optimizing for GPUs

The following techniques can help you get better performance when running
models on GPU hardware using the TensorFlow Lite GPU delegate:

*   **Reshape operations** - Some operations that are quick on a CPU may have a
    high cost for the GPU on mobile devices. Reshape operations are particularly
    expensive to run, including `BATCH_TO_SPACE`, `SPACE_TO_BATCH`,
    `SPACE_TO_DEPTH`, and so forth. You should closely examine use of reshape
    operations, and consider that may have been applied only for exploring data
    or for early iterations of your model. Removing them can significantly
    improve performance.

*   **Image data channels** - On GPU, tensor data is sliced into 4-channels, and
    so a computation on a tensor with the shape `[B,H,W,5]` performs about the
    same on a tensor of shape `[B,H,W,8]`, but significantly worse than
    `[B,H,W,4]`. If the camera hardware you are using supports image frames in
    RGBA, feeding that 4-channel input is significantly faster, since it avoids
    a memory copy from 3-channel RGB to 4-channel RGBX.

*   **Mobile-optimized models** - For best performance, you should consider
    retraining your classifier with a mobile-optimized network architecture.
    Optimization for on-device inferencing can dramatically reduce latency and
    power consumption by taking advantage of mobile hardware features.

## Advanced GPU support

You can use additional, advanced techniques with GPU processing to enable even
better performance for your models, including quantization and serialization.
The following sections describe these techniques in further detail.

### Using quantized models {:#quantized-models}

This section explains how the GPU delegate accelerates 8-bit quantized models,
including the following:

*   Models trained with
    [Quantization-aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)
*   Post-training [dynamic-range quantization](https://www.tensorflow.org/lite/performance/post_training_quant)
*   Post-training [full-integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

To optimize performance, use models that have both floating-point input and
output tensors.

#### How does this work?

Since the GPU backend only supports floating-point execution, we run quantized
models by giving it a ‘floating-point view’ of the original model. At a
high-level, this entails the following steps:

*   *Constant tensors* (such as weights/biases) are de-quantized once into the
    GPU memory. This operation happens when the delegate is enabled for
    TensorFlow Lite.

*   *Inputs and outputs* to the GPU program, if 8-bit quantized, are
    de-quantized and quantized (respectively) for each inference. This operation
    is done on the CPU using TensorFlow Lite’s optimized kernels.

*   *Quantization simulators* are inserted between operations to mimic quantized
    behavior. This approach is necessary for models where ops expect activations
    to follow bounds learnt during quantization.

For information about enabling this feature with the GPU delegate, see the
following:

*   Using [quantized models with GPU on Android](../android/delegates/gpu#quantized-models)
*   Using [quantized models with GPU on iOS](../ios/delegates/gpu#quantized-models)

### Reducing initialization time with serialization {:#delegate_serialization}

The GPU delegate feature allows you to load from pre-compiled kernel code and
model data serialized and saved on disk from previous runs. This approach avoids
re-compilation and can reduce startup time by up to 90%. This improvement is
achieved by exchanging disk space for time savings. You can enable this feature
with a few configurations options, as shown in the following code examples:

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p><pre class="prettyprint lang-cpp">
    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre></p>
    </section>
  </devsite-selector>
</div>

When using the serialization feature, make sure your code complies with these
implementation rules:

*   Store the serialization data in a directory that is not accessible to other
    apps. On Android devices, use
    [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir\(\))
    which points to a location that is private to the current application.
*   The model token must be unique to the device for the specific model. You can
    compute a model token by generating a fingerprint from the model data
    using libraries such as
    [`farmhash::Fingerprint64`](https://github.com/google/farmhash).

Note: Use of this serialization feature requires the
[OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK).
