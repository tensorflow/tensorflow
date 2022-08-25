# TensorFlow Lite in Google Play services (BETA)

TensorFlow Lite is available in the Google Play services API as a public beta on
all Android devices running the current version of Play services. This API lets
you run machine learning (ML) models without statically bundling TensorFlow Lite
libraries into your app, allowing you to:

*   Reduce your app size
*   Gain improved performance from the latest, stable version of TensorFlow Lite

TensorFlow Lite in Google Play services is available through the
[TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary)
and
[TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi).
The Task Library provides optimized out-of-box model interfaces for common
machine learning tasks using visual, audio, and text data. The TensorFlow Lite
Interpreter API, provided by the TensorFlow runtime and support libraries,
provides a more general-purpose interface for building and running ML models.

This page provides a brief overview on how to use the new TensorFlow Lite in
Google Play services APIs in your Android app.

Beta: The functionality described in this page is in beta release. The
functionality and APIs described here may change in future releases.

For more information about Google Play services, see the
[Google Play services](https://developers.google.com/android/guides/overview)
website.

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in
Google Play services, you agree to the <a href="#tos">Terms of Service</a>.
Please read and understand all applicable terms and policies before accessing
the APIs. </aside>

## Add TensorFlow Lite to your app

You can use the TensorFlow Lite in Google Play services API by making a few
changes to your app module dependencies, initializing the new API, and using a
specific class as your interpreter object.

The following instructions provide more details on how to implement the
Interpreter and Task Library APIs in Google Play services. While it is possible
for an app to use both the Interpreter APIs and Task Library APIs, most apps
should only use one set of APIs.

Note: If you are already using TensorFlow Lite in your app, you should review
the [Migrating from stand-alone TensorFlow Lite](#migrating) section.

### Example app with Task Library

You can review and test an example implementation of TensorFlow Lite in Google
Play services in the
[example app](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android_play_services).
This example app uses the Task Library within Play services to create an image
classification app.

### Using the Task Library APIs

The TensorFlow Lite Task API wraps the Interpreter API and provides a high-level
programming interface for common machine learning tasks that use visual, audio,
and text data. You should use the Task API if your application requires one of
the
[supported tasks](../api_docs/java/org/tensorflow/lite/inference_with_metadata/task_library/overview#supported_tasks).

#### 1. Add project dependencies

Your project dependency depends on your machine learning use case. The Task APIs
contain the following libraries:

*   Vision library: `org.tensorflow:tensorflow-lite-task-vision-play-services`
*   Audio library: `org.tensorflow:tensorflow-lite-task-audio-play-services`
*   Text library: `org.tensorflow:tensorflow-lite-task-text-play-services`

Add one of the dependencies to your app project code to access the Play Services
API for TensorFlow Lite. For example, to implement a vision task:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2-beta03'
...
}
```

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
Services API for TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.0-beta03'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.0-beta03'
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
Play Services:

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

### GPU with Task Library APIs

To use the GPU delegate with the Task APIs:

1.  Update the project dependencies to use the GPU delegate from Play Services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0-beta03'
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

1.  Update the project dependencies to use the GPU delegate from Play Services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0-beta03'
    ```

1.  Enable the GPU delegate option in the TFlite initialization:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
          TfLite.initialize(this,
            TfLiteInitializationOptions.builder()
             .setEnableGpuDelegateSupport(true)
             .build())
        </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
          TfLite.initialize(this,
            TfLiteInitializationOptions.builder()
             .setEnableGpuDelegateSupport(true)
             .build());
        </pre>
    </section>
    </devsite-selector>
    </div>

1.  Set GPU delegate in interpreter options to use `DelegateFactory` by calling
    `addDelegateFactory()` within `InterpreterApi.Options()`:

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
    and modify it so that it uses the InterpreterApi.create() call. This new API
    is asynchronous, which means in most cases it's not a drop-in replacement,
    and you must register a listener for when the call completes. Refer to the
    code snippet in [Step 3](#step_3_interpreter) code.
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

## Known issues

After implementing TensorFlow Lite in Google Play services, make sure to test
your application and exercise the machine learning model functions of your app.
If you experience errors or issues you are unable to resolve, please report them
by using the channels outlined in the [Support and feedback](#support) section
below.

### LoadingException: No acceptable module

The `LoadingException` error occurs when an app fails to initialize either the
TFlite or GPU module. This error can occur if you are not part of the
[tflite-play-services-beta-access](https://groups.google.com/g/tflite-play-services-beta-access/about)
group, or when a device is not support GPU hardware acceleration.

#### TFlite module

While testing your app through a development environment during the Beta launch
period, you may get an exception when your app attempts to initialize the
TensorFlow Lite class (`TfLite.intialize(context)`):

```
com.google.android.gms.dynamite.DynamiteModule$LoadingException:
  No acceptable module com.google.android.gms.tflite_dynamite found.
  Local version is 0 and remote version is 0.
```

This error means that the TensorFlow Lite in Google Play services API is not yet
available on your test device. You can resolve this exception by joining this
Google group
[tflite-play-services-beta-access](https://groups.google.com/g/tflite-play-services-beta-access/about)
with *the user account you are using to test on your device.* Once you have been
added to the beta access group, this exception should be resolved. If you are
already part of the beta access group, try restarting the app.

Allow at least one business day after you join this group for access to be
granted and the error to clear. If you continue to experience this error, report
it using the channels outlined in the [Support and feedback](#support) section
below.

Note: This error only occurs when testing this API in a development environment.
Apps that use this API and are installed or updated on a device through the
Google Play Store automatically receive the required libraries.

#### GPU module

Not all devices support hardware acceleration for TFLite. In order to mitigate
errors and potential crashes the download of the Play Services GPU module is
allowlisted only for supported devices.

When initializing TFLite with the GPU delegate, also initialize TFLite with CPU
as a fallback. This allows the app to use CPU with devices that do not support
GPU.

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">
// Instantiate the initialization task with GPU delegate
private var useGpu = true
private val initializeTask: Task&lt;Void> by lazy {
   TfLite.initialize(this,
       TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
       // Add a TFLite initialization task with CPU as a failsafe
       .continueWithTask { task: Task&lt;Void> ->
         if (task.exception != null) {
           useGpu = false
           return@continueWithTask TfLite.initialize(this)
         } else {
           return@continueWithTask Tasks.forResult&lt;Void>(null)
         }
       }
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">
// Initialize TFLite asynchronously
initializeTask = TfLite.initialize(
   this, TfLiteInitializationOptions.builder()
      .setEnableGpuDelegateSupport(true)
      .build()
).continueWithTask(
      task -> {
         if (task.getException() != null) {
            useGpu = false;
            return TfLite.initialize(this);
         } else {
            return Tasks.forResult(null);
         }
      }
);
    </pre>
</section>
</devsite-selector>
</div>

After initializing TFLite with the GPU delegate and a CPU fallback, you can use
the GPU delegate only when it is supported.

**With the Task Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">
lateinit val options
if (useGpu) {
   val baseOptions = BaseOptions.builder().useGpu().build()
   options =
      ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptions)
          .setMaxResults(1)
          .build()
} else {
   options =
      ObjectDetectorOptions.builder()
        .setMaxResults(1)
        .build()
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">
ObjectDetectorOptions options;
   if (useGpu) {
      BaseOptions baseOptions = BaseOptions.builder().useGpu().build()
      options =
        ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptions)
          .setMaxResults(1)
          .build()
   } else {
      options =
         ObjectDetectorOptions.builder()
           .setMaxResults(1)
           .build()
   }
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
val interpreterOptions = if (useGpu) {
  InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(GpuDelegateFactory())
} else {
   InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
}
InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOption)
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">
InterpreterApi.Options options;
  if (useGpu) {
    options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
        .addDelegateFactory(new GpuDelegateFactory());
  } else {
    options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
  }
    </pre>
</section>
</devsite-selector>
</div>

## Limitations

TensorFlow Lite in Google Play services has the following limitations:

*   Support for hardware acceleration delegates is limited to the delegates
    listed in the [Hardware acceleration](#hardware-acceleration) section. No
    other acceleration delegates are supported.
*   Access to TensorFlow Lite via
    [native APIs](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)
    is not supported. Only the TensorFlow Lite Java APIs are available through
    Google Play services.
*   Experimental or deprecated TensorFlow Lite APIs, including custom ops, are
    not supported.

## Support and feedback {:#support}

You can provide feedback and get support through the TensorFlow Issue Tracker.
Please report issues and support requests using the
[Issue template](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md)
for TensorFlow Lite in Google Play services.

## Terms of service {:#tos}

Use of TensorFlow Lite in Google Play services is subject to the
[Google APIs Terms of Service](https://developers.google.com/terms/).

Note: TensorFlow Lite in Google Play services is in beta and, as such, its
functionality as well as associated APIs may change without advance notice.

### Privacy and data collection

When you use TensorFlow Lite in Google Play services APIs, processing of the
input data, such as images, video, text, fully happens on-device, and TensorFlow
Lite in Google Play services does not send that data to Google servers. As a
result, you can use our APIs for processing data that should not leave the
device.

The TensorFlow Lite in Google Play services APIs may contact Google servers from
time to time in order to receive things like bug fixes, updated models and
hardware accelerator compatibility information. The TensorFlow Lite in Google
Play services APIs also sends metrics about the performance and utilization of
the APIs in your app to Google. Google uses this metrics data to measure
performance, debug, maintain and improve the APIs, and detect misuse or abuse,
as further described in our
[Privacy Policy](https://policies.google.com/privacy).

**You are responsible for informing users of your app about Google's processing
of TensorFlow Lite in Google Play services metrics data as required by
applicable law.**

Data we collect includes the following:

+   Device information (such as manufacturer, model, OS version and build) and
    available ML hardware accelerators (GPU and DSP). Used for diagnostics and
    usage analytics.
+   Device identifier used for diagnostics and usage analytics.
+   App information (package name, app version). Used for diagnostics and usage
    analytics.
+   API configuration (such as which delegates are being used). Used for
    diagnostics and usage analytics.
+   Event type (such as interpreter creation, inference). Used for diagnostics
    and usage analytics.
+   Error codes. Used for diagnostics.
+   Performance metrics. Used for diagnostics.

## Next steps

For more information about implementing machine learning in your mobile
application with TensorFlow Lite, see the
[TensorFlow Lite Developer Guide](https://www.tensorflow.org/lite/guide). You
can find additional TensorFlow Lite models for image classification, object
detection, and other applications on the
[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite).
