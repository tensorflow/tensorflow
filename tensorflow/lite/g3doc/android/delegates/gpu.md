# GPU acceleration delegate for Android

Using graphics processing units (GPUs) to run your machine learning (ML) models
can dramatically improve the performance of your model and the user experience
of your ML-enabled applications. On Android devices, you can enable use of
GPU-accelerated execution of your models using a [*delegate*](../../performance/delegates).
Delegates act as hardware drivers for TensorFlow Lite, allowing you to run
the code of your model on GPU processors.

This page describes how to enable GPU acceleration for TensorFlow Lite models
in Android apps. For more information about using the GPU delegate for
TensorFlow Lite, including best practices and advanced techniques, see the
[GPU delegates](../../performance/gpu) page.

## Use GPU with Task Library APIs

The TensorFlow Lite
[Task Libraries](../../inference_with_metadata/task_library/overview) provide a
set of task-specific APIs for building a machine learning applications. This
section describes how to use the GPU accelerator delegate with these APIs.

### Add project dependencies

Enable access to the GPU delegate APIs with the TensorFlow Lite Task Libraries
by adding the following dependencies update your development projects
`build.gradle` file to include the `tensorflow-lite-gpu-delegate-plugin` package
as shown in the following code example:

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Enable GPU acceleration

Enable the GPU delegate option for your Task API model class with the
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

## Use GPU with Interpreter API

The TensorFlow Lite [Interpreter API](../../api_docs/java/org/tensorflow/lite/InterpreterApi)
provides a set of general purpose APIs for building a machine learning
applications. This section describes how to use the GPU accelerator delegate
with these APIs.

### Add project dependencies

Enable access to the GPU delegate APIs by adding the following dependencies
update your development projects `build.gradle` file to include the
`org.tensorflow:tensorflow-lite-gpu` package as shown in the following code
example:

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu'
}
```

### Enable GPU acceleration

Then run TensorFlow Lite on GPU with `TfLiteDelegate`. In Java, you can specify
the `GpuDelegate` through `Interpreter.Options`.

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

Note: The GPU delegate must be created on the same thread that runs it.
Otherwise, you may see the following error, `TfLiteGpuDelegate Invoke:
GpuDelegate must run on the same thread where it was initialized.`

The GPU delegate can also be used with ML model binding in Android Studio.
For more information, see
[Generate model interfaces using metadata](../../inference_with_metadata/codegen#acceleration).

## Advanced GPU support

This section covers advanced uses of the GPU delegate for Android, including
the C API, C++ API, and use of quantized models.

### C/C++ API for Android

Use the TensorFlow Lite GPU delegate for Android in C or C++ by creating the
delegate with `TfLiteGpuDelegateV2Create()` and destroying it with
`TfLiteGpuDelegateV2Delete()`, as shown in the following example code:

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

Review the `TfLiteGpuDelegateOptionsV2` object code to build a delegate instance
with custom options. You can initialize the default options with
`TfLiteGpuDelegateOptionsV2Default()` and then modify them as necessary.

The TensorFlow Lite GPU delegate for Android in C or C++ uses the
[Bazel](https://bazel.io) build system. You can build the delegate using the
following command:

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

When calling `Interpreter::ModifyGraphWithDelegate()` or
`Interpreter::Invoke()`, the caller must have an `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`. If
an `EGLContext` does not exist, the delegate creates one internally, but then
you must ensure that `Interpreter::Invoke()` is always called from the same
thread in which `Interpreter::ModifyGraphWithDelegate()` was called.

### Quantized models {:#quantized-models}

Android GPU delegate libraries support quantized models by default. You do not
have to make any code changes to use quantized models with the GPU delegate. The
following section explains how to disable quantized support for testing or
experimental purposes.

#### Disable quantized model support

The following code shows how to ***disable*** support for quantized models.

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre></p>
    </section>
    <section>
      <h3>C++</h3>
      <p><pre class="prettyprint lang-c++">
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre></p>
    </section>
  </devsite-selector>
</div>

For more information about running quantized models with GPU acceleration,
see [GPU delegate](../../performance/gpu#quantized-models) overview.
