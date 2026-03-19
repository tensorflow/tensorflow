# TensorFlow Lite in Google Play services

TensorFlow Lite is available in Google Play services runtime for all Android
devices running the current version of Play services. This runtime allows you to
run machine learning (ML) models without statically bundling TensorFlow Lite
libraries into your app.

With the Google Play services API, you can reduce the size of your apps and gain
improved performance from the latest stable version of the libraries. TensorFlow
Lite in Google Play services is the recommended way to use TensorFlow Lite on
Android.

You can get started with the Play services runtime with the
[Quickstart](../android/quickstart), which provides a step-by-step guide to
implement a sample application. If you are already using stand-alone TensorFlow
Lite in your app, refer to the
[Migrating from stand-alone TensorFlow Lite](#migrating) section to update an
existing app to use the Play services runtime. For more information about Google
Play services, see the
[Google Play services](https://developers.google.com/android/guides/overview)
website.

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in
Google Play services APIs, you agree to the <a href="#tos">Terms of Service</a>.
Please read and understand all applicable terms and policies before accessing
the APIs. </aside>

## Using the Play services runtime

The TensorFlow Lite in Google Play services is available through the following
programming language apis:

-   Java API - [see guide](../android/java)
-   C API - [see guide](../android/native)

## Limitations

TensorFlow Lite in Google Play services has the following limitations:

*   Support for hardware acceleration delegates is limited to the delegates
    listed in the [Hardware acceleration](#hardware-acceleration) section. No
    other acceleration delegates are supported.
*   Experimental or deprecated TensorFlow Lite APIs, including custom ops, are
    not supported.

## Support and feedback {:#support}

You can provide feedback and get support through the TensorFlow Issue Tracker.
Please report issues and support requests using the
[Issue template](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md)
for TensorFlow Lite in Google Play services.

## Terms of service {:#tos}

Use of TensorFlow Lite in Google Play services APIs is subject to the
[Google APIs Terms of Service](https://developers.google.com/terms/).

### Privacy and data collection

When you use TensorFlow Lite in Google Play services APIs, processing of the
input data, such as images, video, text, fully happens on-device, and TensorFlow
Lite in Google Play services APIs does not send that data to Google servers. As
a result, you can use our APIs for processing data that should not leave the
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
of TensorFlow Lite in Google Play services APIs metrics data as required by
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
