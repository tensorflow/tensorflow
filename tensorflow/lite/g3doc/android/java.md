# TensorFlow Lite in Google Play services Java API

TensorFlow Lite in Google Play services can also be accessed using Java APIs, in
addition to the Native API. In particular, TensorFlow Lite in Google Play
services is available through the
[TensorFlow Lite Task API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/package-summary)
and the
[TensorFlow Lite Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).
The Task Library provides optimized out-of-the-box model interfaces for common
machine learning tasks using visual, audio, and text data. The TensorFlow Lite
Interpreter API, provided by the TensorFlow runtime, provides a more
general-purpose interface for building and running ML models.

The following sections provide instructions on how to use the Interpreter and
Task Library APIs with TensorFlow Lite in Google Play services. While it is
possible for an app to use both the Interpreter APIs and Task Library APIs, most
apps should only use one set of APIs.

### Using the Task Library APIs

The TensorFlow Lite Task API wraps the Interpreter API and provides a high-level
programming interface for common machine learning tasks that use visual, audio,
and text data. You should use the Task API if your application requires one of
the
[supported tasks](../inference_with_metadata/task_library/overview.md#supported-tasks).

#### 1. Add project dependencies

Your project dependency depends on your machine learning use case. The Task APIs
contain the following libraries:

*   Vision library: `org.tensorflow:tensorflow-lite-task-vision-play-services`
*   Audio library: `org.tensorflow:tensorflow-lite-task-audio-play-services`
*   Text library: `org.tensorflow:tensorflow-lite-task-text-play-services`

Add one of the dependencies to your app project code to access the Play services
API for TensorFlow Lite. For example, use the following to implement a vision
task:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

Caution: The TensorFlow Lite Tasks Audio library version 0.4.2 maven repository
is incomplete. Use version 0.4.2.1 for this library instead:
`org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`.

#### 2. Add initialization of TensorFlow Lite

Initialize the TensorFlow Lite component of the Google Play services API
*before* using the TensorFlow Lite APIs. The following example initializes the
vision library:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
init {
  TfLiteVision.initialize(context)
}
</pre>
    </section>
  </devsite-selector>
</div>

Important: Make sure the `TfLite.initialize` task completes before executing
code that accesses TensorFlow Lite APIs.

Tip: The TensorFlow Lite modules are installed at the same time your application
is installed or updated from the Play Store. You can check the availability of
the modules by using `ModuleInstallClient` from the Google Play services APIs.
For more information on checking module availability, see
[Ensuring API availability with ModuleInstallClient](https://developers.google.com/android/guides/module-install-apis).

#### 3. Run inferences

After initializing the TensorFlow Lite component, call the `detect()` method to
generate inferences. The exact code within the `detect()` method varies
depending on the library and use case. The following is for a simple object
detection use case with the `TfLiteVision` library:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
fun detect(...) {
  if (!TfLiteVision.isInitialized()) {
    Log.e(TAG, "detect: TfLiteVision is not initialized yet")
    return
  }

  if (objectDetector == null) {
    setupObjectDetector()
  }

  ...

}
</pre>
    </section>
  </devsite-selector>
</div>

Depending on the data format, you may also need to preprocess and convert your
data within the `detect()` method before generating inferences. For example,
image data for an object detector requires the following:

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Using the Interpreter APIs

The Interpreter APIs offer more control and flexibility than the Task Library
APIs. You should use the Interpreter APIs if your machine learning task is not
supported by the Task library, or if you require a more general-purpose
interface for building and running ML models.

#### 1. Add project dependencies

Add the following dependencies to your app project code to access the Play
services API for TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.4.0'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.4.0'
...
}
```

#### 2. Add initialization of TensorFlow Lite

Initialize the TensorFlow Lite component of the Google Play services API
*before* using the TensorFlow Lite APIs:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
val initializeTask: Task&lt;Void> by lazy { TfLite.initialize(this) }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
Task&lt;Void> initializeTask = TfLite.initialize(context);
</pre>
    </section>
  </devsite-selector>
</div>

Note: Make sure the `TfLite.initialize` task completes before executing code
that accesses TensorFlow Lite APIs. Use the `addOnSuccessListener()` method, as
shown in the next section.

#### 3. Create an Interpreter and set runtime option {:#step_3_interpreter}

Create an interpreter using `InterpreterApi.create()` and configure it to use
Google Play services runtime, by calling `InterpreterApi.Options.setRuntime()`,
as shown in the following example code:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private lateinit var interpreter: InterpreterApi
...
initializeTask.addOnSuccessListener {
  val interpreterOption =
    InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  interpreter = InterpreterApi.create(
    modelBuffer,
    interpreterOption
  )}
  .addOnFailureListener { e ->
    Log.e("Interpreter", "Cannot initialize interpreter", e)
  }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private InterpreterApi interpreter;
...
initializeTask.addOnSuccessListener(a -> {
    interpreter = InterpreterApi.create(modelBuffer,
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY));
  })
  .addOnFailureListener(e -> {
    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s",
          e.getMessage()));
  });
</pre>
    </section>
  </devsite-selector>
</div>

You should use the implementation above because it avoids blocking the Android
user interface thread. If you need to manage thread execution more closely, you
can add a `Tasks.await()` call to interpreter creation:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
import androidx.lifecycle.lifecycleScope
...
lifecycleScope.launchWhenStarted { // uses coroutine
  initializeTask.await()
}
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
@BackgroundThread
InterpreterApi initializeInterpreter() {
    Tasks.await(initializeTask);
    return InterpreterApi.create(...);
}
</pre>
    </section>
  </devsite-selector>
</div>

Warning: Do not call `.await()` on the foreground user interface thread because
it interrupts display of user interface elements and creates a poor user
experience.

#### 4. Run inferences

Using the `interpreter` object you created, call the `run()` method to generate
an inference.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
interpreter.run(inputBuffer, outputBuffer)
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
interpreter.run(inputBuffer, outputBuffer);
</pre>
    </section>
  </devsite-selector>
</div>

## Hardware acceleration {:#hardware-acceleration}

TensorFlow Lite allows you to accelerate the performance of your model using
specialized hardware processors, such as graphics processing units (GPUs). You
can take advantage of these specialized processors using hardware drivers called
[*delegates*](https://www.tensorflow.org/lite/performance/delegates). You can
use the following hardware acceleration delegates with TensorFlow Lite in Google
Play services:

-   *[GPU delegate](https://www.tensorflow.org/lite/performance/gpu)
    (recommended)* - This delegate is provided through Google Play services and
    is dynamically loaded, just like the Play services versions of the Task API
    and Interpreter API.

-   [*NNAPI delegate*](https://www.tensorflow.org/lite/android/delegates/nnapi) -
    This delegate is available as an included library dependency in your Android
    development project, and is bundled into your app.

For more information about hardware acceleration with TensorFlow Lite, see the
[TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates)
page.

### Checking device compatibility

Not all devices support GPU hardware acceleration with TFLite. In order to
mitigate errors and potential crashes, use the
`TfLiteGpu.isGpuDelegateAvailable` method to check whether a device is
compatible with the GPU delegate.

Use this method to confirm whether a device is compatible with GPU, and use CPU
or the NNAPI delegate as a fallback for when GPU is not supported.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

Once you have a variable like `useGpuTask`, you can use it to determine whether
devices use the GPU delegate. The following examples show how this can be done
with both the Task Library and Interpreter APIs.

**With the Task Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">
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
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">
Task&lt;ObjectDetectorOptions> optionsTask = useGpuTask.continueWith({ task ->
  BaseOptions baseOptionsBuilder = BaseOptions.builder();
  if (task.getResult()) {
    baseOptionsBuilder.useGpu();
  }
  return ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptionsBuilder.build())
          .setMaxResults(1)
          .build()
});
    </pre>
</section>
</devsite-selector>
</div>

**With the Interpreter Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">
val interpreterTask = useGpuTask.continueWith { task ->
  val interpreterOptions = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  if (task.result) {
      interpreterOptions.addDelegateFactory(GpuDelegateFactory())
  }
  InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOptions)
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">
Task&lt;InterpreterApi.Options> interpreterOptionsTask = useGpuTask.continueWith({ task ->
  InterpreterApi.Options options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
  if (task.getResult()) {
     options.addDelegateFactory(new GpuDelegateFactory());
  }
  return options;
});
    </pre>
</section>
</devsite-selector>
</div>

### GPU with Task Library APIs

To use the GPU delegate with the Task APIs:

1.  Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.4.0'
    ```

1.  Initialize the GPU delegate with `setEnableGpuDelegateSupport`. For example,
    you can initialize the GPU delegate for `TfLiteVision` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
        </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build());
        </pre>
    </section>
    </devsite-selector>
    </div>

1.  Enable the GPU delegate option with
    [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder):

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
        val baseOptions = BaseOptions.builder().useGpu().build()
        </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
        BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
        </pre>
    </section>
    </devsite-selector>
    </div>

1.  Configure the options using `.setBaseOptions`. For example, you can set up
    GPU in `ObjectDetector` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
        val options =
            ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setMaxResults(1)
                .build()
        </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
        ObjectDetectorOptions options =
            ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setMaxResults(1)
                .build();
        </pre>
    </section>
    </devsite-selector>
    </div>

### GPU with Interpreter APIs

To use the GPU delegate with the Interpreter APIs:

1.  Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.4.0'
    ```

1.  Enable the GPU delegate option in the TFlite initialization:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
        TfLite.initialize(context,
          TfLiteInitializationOptions.builder()
           .setEnableGpuDelegateSupport(true)
           .build())
        </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
        TfLite.initialize(context,
          TfLiteInitializationOptions.builder()
           .setEnableGpuDelegateSupport(true)
           .build());
        </pre>
    </section>
    </devsite-selector>
    </div>

1.  Enable GPU delegate in the interpreter options: set the delegate factory to
    GpuDelegateFactory by calling `addDelegateFactory()
    within`InterpreterApi.Options()`:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
        val interpreterOption = InterpreterApi.Options()
         .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
         .addDelegateFactory(GpuDelegateFactory())
        </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
        Options interpreterOption = InterpreterApi.Options()
          .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
          .addDelegateFactory(new GpuDelegateFactory());
        </pre>
    </section>
    </devsite-selector>
    </div>

## Migrating from stand-alone TensorFlow Lite {:#migrating}

If you are planning to migrate your app from stand-alone TensorFlow Lite to the
Play services API, review the following additional guidance for updating your
app project code:

1.  Review the [Limitations](#limitations) section of this page to ensure your
    use case is supported.
2.  Prior to updating your code, do performance and accuracy checks for your
    models, particularly if you are using versions of TensorFlow Lite earlier
    than version 2.1, so you have a baseline to compare against the new
    implementation.
3.  If you have migrated all of your code to use the Play services API for
    TensorFlow Lite, you should remove the existing TensorFlow Lite *runtime
    library* dependencies (entries with
    <code>org.tensorflow:**tensorflow-lite**:*</code>) from your build.gradle
    file so that you can reduce your app size.
4.  Identify all occurrences of `new Interpreter` object creation in your code,
    and modify each one so that it uses the InterpreterApi.create() call. The
    new TfLite.initialize is asynchronous, which means in most cases it's not a
    drop-in replacement: you must register a listener for when the call
    completes. Refer to the code snippet in [Step 3](#step_3_interpreter) code.
5.  Add `import org.tensorflow.lite.InterpreterApi;` and `import
    org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` to any source
    files using the `org.tensorflow.lite.Interpreter` or
    `org.tensorflow.lite.InterpreterApi` classes.
6.  If any of the resulting calls to `InterpreterApi.create()` have only a
    single argument, append `new InterpreterApi.Options()` to the argument list.
7.  Append `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` to the last argument of
    any calls to `InterpreterApi.create()`.
8.  Replace all other occurrences of the `org.tensorflow.lite.Interpreter` class
    with `org.tensorflow.lite.InterpreterApi`.

If you want to use stand-alone TensorFlow Lite and the Play services API
side-by-side, you must use TensorFlow Lite 2.9 (or later). TensorFlow Lite 2.8
and earlier versions are not compatible with the Play services API version.
