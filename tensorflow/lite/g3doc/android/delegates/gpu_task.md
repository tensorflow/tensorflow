# GPU acceleration delegate with Task library

Using graphics processing units (GPUs) to run your machine learning (ML) models
can dramatically improve the performance and the user experience
of your ML-enabled applications. On Android devices, you can enable
GPU-accelerated execution of your models using a
[*delegate*](../../performance/delegates) and one of the following APIs:

- Interpreter API - [guide](./gpu)
- Task library API - this guide
- Native (C/C++) API - this [guide](./gpu_native)

This page describes how to enable GPU acceleration for TensorFlow Lite models in
Android apps using the Task library.
For more information about the GPU delegate for TensorFlow Lite,
including best practices and advanced techniques, see the
[GPU delegates](../../performance/gpu) page.

## Use GPU with TensorFlow Lite with Google Play services

The TensorFlow Lite
[Task Libraries](../../inference_with_metadata/task_library/overview) provide a
set of task-specific APIs for building machine learning applications. This
section describes how to use the GPU accelerator delegate with these APIs using
TensorFlow Lite with Google Play services.

[TensorFlow Lite with Google Play services](../play_services) is the recommended
path to use TensorFlow Lite on Android. If your application is targeting devices
not running Google Play, see the
[GPU with Task Library and standalone TensorFlow Lite](#standalone)
section.

### Add project dependencies

To enable access to the GPU delegate with the TensorFlow Lite Task
Libraries using Google Play services, add
`com.google.android.gms:play-services-tflite-gpu` to the
dependencies of your app's `build.gradle` file:

```
dependencies {
  ...
  implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
}
```

### Enable GPU acceleration

Then, verify asynchronously that GPU delegate is available for the device using
the
[`TfLiteGpu`](https://developers.google.com/android/reference/com/google/android/gms/tflite/gpu/support/TfLiteGpu)
class and enable the GPU delegate option for your Task API model class with the
[`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder)
class. For example, you can set up GPU in `ObjectDetector` as shown in the
following code examples:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p><pre class="prettyprint lang-kotlin">
        val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

        lateinit val optionsTask = useGpuTask.continueWith { task ->
          val baseOptionsBuilder = BaseOptions.builder()
          if (task.result) {
            baseOptionsBuilder.useGpu()
          }
        ObjectDetectorOptions.builder()
                  .setBaseOptions(baseOptionsBuilder.build())
                  .setMaxResults(1)
                  .build()
        }
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
      Task<Boolean> useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

      Task<ObjectDetectorOptions> optionsTask = useGpuTask.continueWith({ task ->
        BaseOptions baseOptionsBuilder = BaseOptions.builder();
        if (task.getResult()) {
          baseOptionsBuilder.useGpu();
        }
        return ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setMaxResults(1)
                .build()
      });
      </pre></p>
    </section>
  </devsite-selector>
</div>

## Use GPU with standalone TensorFlow Lite {:#standalone}

If your application is targets devices which are not running Google Play,
it is possible to bundle the GPU delegate to your application and use it
with the standalone version of TensorFlow Lite.

### Add project dependencies

To enable access to the GPU delegate with the TensorFlow Lite Task
Libraries using the standalone version of TensorFlow Lite, add
`org.tensorflow:tensorflow-lite-gpu-delegate-plugin` to the
dependencies of your app's `build.gradle` file:

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite'
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Enable GPU acceleration

Then enable the GPU delegate option for your Task API model class with the
[`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder)
class. For example, you can set up GPU in `ObjectDetector` as shown in the
following code examples:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p><pre class="prettyprint lang-kotlin">
    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    val baseOptions = BaseOptions.builder().useGpu().build()

    val options =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build()

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options)
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    BaseOptions baseOptions = BaseOptions.builder().useGpu().build();

    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build();

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options);
      </pre></p>
    </section>
  </devsite-selector>
</div>

