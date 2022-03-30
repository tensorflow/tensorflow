# TensorFlow Lite in Google Play services (BETA)

TensorFlow Lite is available in the Google Play services API as a public beta on
all Android devices running the current version of Play services, starting
February 17, 2022. This API lets you run machine learning models without
statically bundling TensorFlow Lite libraries into your app, allowing you to:

*   Reduce your app size
*   Gain improved performance from the latest, stable version of TensorFlow Lite

This page provides a brief overview on how to use the new TensorFlow Lite in
Google Play services APIs in your Android app.

Beta: The functionality described in this page is in beta release. The
functionality and APIs described here may change in future releases.

For more information about Google Play services, see the
[Google Play services](https://developers.google.com/android/guides/overview)
website.


## Add TensorFlow Lite to your app

You can use the TensorFlow Lite in Google Play services API by making a few
changes to your app module dependencies, initializing the new API, and using a
specific class as your interpreter object. The following instructions provide
more details on how to modify your app code.

Note: If you are already using TensorFlow Lite in your app, you should review
the [Migrating from Standalone TensorFlow Lite](#migrating) section.


### 1. Add project dependencies

Add the following dependencies to your app project code to access the Play
Services API for TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.0-beta01'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.0-beta01'
...
}
```

Also, add the following maven dependency to your app project code:

```
maven {
  name 'ossrh-snapshot'
  url 'https://oss.sonatype.org/content/repositories/snapshots'
}
```

Note: You should only include this dependency during the Beta launch period.


### 2. Add initialization of TensorFlow Lite

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

### 3. Create an Interpreter and set runtime option {:#step_3}

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
private lateinit val interpreter: InterpreterApi
...
initializeTask.addOnSuccessListener(object: OnSuccessListener {
     override fun onSuccess(response: DataReadResponse) {
       interpreter = InterpreterApi.create(modelBuffer,
         object: InterpreterApi.Options().setRuntime(
                    InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY))
     }
   })
   .addOnFailureListener(object: OnFailureListener {
     override fun onFailure(ex: Exception) {
       Log.e("Interpreter", "Cannot initialize interpreter", ex)
     }
   })
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


### 4. Run inferences

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



Important: While testing your app from a development environment during the Beta
launch period, you may get an exception when your app attempts to initialize the
TensorFlow Lite class (`TfLite.intialize(context)`). This means that the Google
Play services API is not yet available on your test device. For information on
resolving this issue, see the [Testing](#testing) section.


## TensorFlow Lite in Google Play services

The TensorFlow Lite in Google Play services API lets you access the actual
[TensorFlow Lite Java API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/package-summary)
in your app after you initialize it using a new method in Play services. This
approach keeps you keep code changes for existing apps to a minimum, and
simplifies new implementations. For more information about the API for this
feature, see the
[TensorFlow Lite API reference](https://developers.google.com/android/reference/com/google/android/gms/tflite/java/package-summary)


## Migrating from Standalone TensorFlow Lite {:#migrating}

If you are planning to migrate your app from standalone TensorFlow Lite to the
Play services API, review the following additional guidance for updating
your app project code:

1.  Review the [Limitations](#limitations) section of this page to ensure your
    use case is supported.
2.  Prior to updating your code, do performance and accuracy checks for your
    models, particularly if you are using versions of TensorFlow Lite earlier
    than version 2.1, so you have a baseline to compare against the new
    implementation.
3.  If you have migrated all of your code to use the Play services API for
    TensorFlow Lite, you should remove the existing TensorFlow Lite *runtime
    library* dependencies (entries with `org.tensorflow:**tensorflow-lite**:\*`)
    from your build.gradle file so that you can reduce your app size.
4.  Identify all occurrences of `new Interpreter` object creation in your code,
    and modify it so that it uses the InterpreterApi.create() call. This new API
    is asynchronous, which means in most cases it's not a drop-in replacement,
    and you must register a listener for when the call completes. Refer to the
    code snippet in [Step 3](#step_3) code.
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

If you want to use standalone TensorFlow Lite and the Play services API
side-by-side, you must link to the nightly build version of TensorFlow Lite.
TensorFlow Lite 2.8 and earlier versions are not compatible with the Play
services API version.


## Example App

You can review and test an example implementation of TensorFlow Lite in Google
Play services in the
[example app](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android_play_services).

## Testing

After implementing TensorFlow Lite in Google Play services, make sure
to test your application and exercise the machine learning model functions of
your app. If you experience errors or issues you are unable to resolve, please
report them by using the channels outlined in the
[Support and feedback](#support) section below.


### LoadingException: No acceptable module

While testing your app through a development environment during the Beta launch
period, you may get an exception when your app attempts to initialize the
TensorFlow Lite class (`TfLite.intialize(context)`):

```
com.google.android.gms.dynamite.DynamiteModule$LoadingException:
  No acceptable module com.google.android.gms.tflite_dynamite found.
  Local version is 0 and remote version is 0.
```

This means that the TensorFlow Lite in Google Play services API is not yet
available on your test device. You can resolve this exception by requesting
access to the access group for the beta API. Fill out the
[form to request access](https://services.google.com/fb/forms/tflite-play-manual-access/),
and *be sure to provide the user account that you are using to test your
device*. Once you have been added to the beta access group, this exception
should be resolved.

Allow at least one business day for the request to be received and access
granted. If you continue to experience this error, report it using the channels
outlined in the [Support and feedback](#support) section below.

Note: This error only occurs when testing this API in a development environment.
Apps that use this API and are installed or updated on a device through the
Google Play Store automatically receive the required libraries.


## Limitations

TensorFlow Lite in Google Play services is currently at public beta and has the
following limitations:

*   Only the [NNAPI Delegate](https://www.tensorflow.org/lite/performance/nnapi)
    is currently supported by Google Play services. Other TensorFlow Lite
    [Delegates](https://www.tensorflow.org/lite/performance/delegates),
    including GPU, XNNPack, and Flex are not currently supported.
*   Access to TensorFlow Lite via
    [native APIs](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)
    is not supported. Only the TensorFlow Lite Java APIs are available through
    Google Play services.
*   Experimental or deprecated TensorFlow Lite APIs, including custom ops, are
    not supported.


## Support and feedback {:#support}

You can provide feedback and get support for this beta release feature, through
the TensorFlow Issue Tracker. Please report issues and support requests using
the
[Issue template](https://github.com/tensorflow/tensorflow/issues/new?assignees=&labels=type%3Aothers&template=100-tflite-in-play-services.md)
for TensorFlow Lite in Google Play services.


## Terms and Privacy Policy

Use of TensorFlow Lite in Google Play services is subject to the [Google APIs
Terms of Service](https://developers.google.com/terms/).
Note that TensorFlow Lite in Google Play services is in beta and, as such, its
functionality as well as associated APIs may change without advance notice.

When you use TensorFlow Lite in Google Play services APIs, processing of the
input data, such as images, video, text, fully happens on-device, and TensorFlow
Lite in Google Play services does not send that data to Google servers. As a
result, you can use our APIs for processing data that should not leave the
device.

The TensorFlow Lite in Google Play services APIs may contact Google servers
from time to time in order to receive things like bug fixes, updated models and
hardware accelerator compatibility information. The TensorFlow Lite in Google
Play services APIs also sends metrics about the performance and utilization of
the APIs in your app to Google. Google uses this metrics data to measure
performance, debug, maintain and improve the APIs, and detect misuse or abuse,
as further described in our [Privacy
Policy](https://policies.google.com/privacy).

**You are responsible for informing users of your app about Google's processing
of TensorFlow Lite in Google Play services metrics data as required by
applicable law.**

Data we collect includes the following:

+   Device information (such as manufacturer, model, OS version and
    build) and available ML hardware accelerators (GPU and DSP). Used for
    diagnostics and usage analytics.
+   Device identifier used for diagnostics and usage analytics.
+   App information (package name, app version). Used for diagnostics and
    usage analytics.
+   API configuration (such as which delegates are being used). Used for
    diagnostics and usage analytics.
+   Event type (such as interpreter creation, inference). Used for
    diagnostics and usage analytics.
+   Error codes. Used for diagnostics.
+   Performance metrics. Used for diagnostics.


## Next steps

For more information about implementing machine learning in your mobile
application with TensorFlow Lite, see the
[TensorFlow Lite Developer Guide](https://www.tensorflow.org/lite/guide). You
can find additional TensorFlow Lite models for image classification, object
detection, and other applications on the TensorFlow Lite
[Model library](https://www.tensorflow.org/lite/guide/hosted_models) page.
