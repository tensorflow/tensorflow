# GPU acceleration delegate with Interpreter API

Using graphics processing units (GPUs) to run your machine learning (ML) models
can dramatically improve the performance and the user experience
of your ML-enabled applications. On Android devices, you can enable

[*delegate*](../../performance/delegates) and one of the following APIs:

- Interpreter API - this guide
- Task library API - [guide](./gpu_task)
- Native (C/C++) API - [guide](./gpu_native)

This page describes how to enable GPU acceleration for TensorFlow Lite models in
Android apps using the Interpreter API.
For more information about using the GPU delegate for
TensorFlow Lite, including best practices and advanced techniques, see the
[GPU delegates](../../performance/gpu) page.

## Use GPU with TensorFlow Lite with Google Play services

The TensorFlow Lite [Interpreter API](https://tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi)
provides a set of general purpose APIs for building a machine learning
applications. This section describes how to use the GPU accelerator delegate
with these APIs with TensorFlow Lite with Google Play services.

[TensorFlow Lite with Google Play services](../play_services) is the recommended
path to use TensorFlow Lite on Android. If your application is targeting devices
not running Google Play, see the
[GPU with Interpreter API and standalone TensorFlow Lite](#standalone)
section.

### Add project dependencies

To enable access to the GPU delegate, add
`com.google.android.gms:play-services-tflite-gpu` to your app's `build.gradle`
file:

```
dependencies {
    ...
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
}
```

### Enable GPU acceleration

Then initialize TensorFlow Lite with Google Play services with the GPU support:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p><pre class="prettyprint lang-kotlin">
    val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

    val interpreterTask = useGpuTask.continueWith { useGpuTask ->
      TfLite.initialize(context,
          TfLiteInitializationOptions.builder()
          .setEnableGpuDelegateSupport(useGpuTask.result)
          .build())
      }
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">
    Task<boolean> useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

    Task<Options> interpreterOptionsTask = useGpuTask.continueWith({ task ->
      TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build());
    });
      </pre></p>
    </section>
  </devsite-selector>
</div>

You can finally initialize the interpreter passing a `GpuDelegateFactory`
through `InterpreterApi.Options`:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p><pre class="prettyprint lang-kotlin">

    val options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(GpuDelegateFactory())

    val interpreter = InterpreterApi(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre></p>
    </section>
    <section>
      <h3>Java</h3>
      <p><pre class="prettyprint lang-java">

    Options options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(new GpuDelegateFactory());

    Interpreter interpreter = new InterpreterApi(model, options);

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

## Use GPU with standalone TensorFlow Lite {:#standalone}

If your application is targets devices which are not running Google Play,
it is possible to bundle the GPU delegate to your application and use it
with the standalone version of TensorFlow Lite.

### Add project dependencies

To enable access to the GPU delegate, add
`org.tensorflow:tensorflow-lite-gpu-delegate-plugin` to your app's `build.gradle`
file:

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
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
  </devsite-selector>
</div>

For more information about running quantized models with GPU acceleration,
see [GPU delegate](../../performance/gpu#quantized-models) overview.