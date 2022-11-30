# TensorFlow Lite for Android

TensorFlow Lite lets you run TensorFlow machine learning (ML) models in your
Android apps. The TensorFlow Lite system provides prebuilt and customizable
execution environments for running models on Android quickly and efficiently,
including options for hardware acceleration.

## Learning roadmap {:.hide-from-toc}

<section class="devsite-landing-row devsite-landing-row-3-up devsite-landing-row-100" header-position="top">
<div class="devsite-landing-row-inner">
<div class="devsite-landing-row-group">
  <div class="devsite-landing-row-item devsite-landing-row-item-no-media tfo-landing-page-card" description-position="bottom">
    <div class="devsite-landing-row-item-description">
    <div class="devsite-landing-row-item-body">
    <div class="devsite-landing-row-item-description-content">
      <a href="#machine_learning_models">
      <h3 class="no-link hide-from-toc" id="code-design" data-text="Code design">Code design</h3></a>
      Learn concepts and code design for building Android apps with TensorFlow
      Lite, just <a href="#machine_learning_models">keep reading</a>.
    </div>
    </div>
    </div>
  </div>
  <div class="devsite-landing-row-item devsite-landing-row-item-no-media tfo-landing-page-card" description-position="bottom">
    <div class="devsite-landing-row-item-description">
    <div class="devsite-landing-row-item-body">
    <div class="devsite-landing-row-item-description-content">
      <a href="./quickstart">
      <h3 class="no-link hide-from-toc" id="coding-quickstart" data-text="Coding Quickstart">Code Quickstart</h3></a>
      Start coding an Android app with TensorFlow Lite right away with the
      <a href="./quickstart">Quickstart</a>.
    </div>
    </div>
    </div>
  </div>
  <div class="devsite-landing-row-item devsite-landing-row-item-no-media tfo-landing-page-card" description-position="bottom">
    <div class="devsite-landing-row-item-description">
    <div class="devsite-landing-row-item-body">
    <div class="devsite-landing-row-item-description-content">
      <a href="../models">
      <h3 class="no-link hide-from-toc" id="ml-models" data-text="ML models">ML models</h3></a>
      Learn about choosing and using ML models with TensorFlow Lite, see the
      <a href="../models">Models</a> docs.
    </div>
    </div>
    </div>
  </div>
</div>
</div>
</section>

## Machine learning models

TensorFlow Lite uses TensorFlow models that are converted into a smaller,
portable, more efficient machine learning model format. You can use pre-built
models with TensorFlow Lite on Android, or build your own TensorFlow models and
convert them to TensorFlow Lite format.

**Key Point:** TensorFlow Lite models and TensorFlow models have a *different
format and are not interchangeable.* TensorFlow models can be converted into the
TensorFlow Lite models, but that process is not reversible.

This page discusses using already-built machine learning models and does not
cover building, training, testing, or converting models. Learn more about
picking, modifying, building, and converting machine learning models for
TensorFlow Lite in the [Models](../models) section.

## Run models on Android

A TensorFlow Lite model running inside an Android app takes in data, processes
the data, and generates a prediction based on the model's logic. A TensorFlow
Lite model requires a special runtime environment in order to execute, and the
data that is passed into the model must be in a specific data format, called a
[*tensor*](../../guide/tensor). When a model processes the data, known as running
an *inference*, it generates prediction results as new tensors, and passes them
to the Android app so it can take action, such as showing the result to a user
or executing additional business logic.

![Functional execution flow for TensorFlow Lite models in Android
apps](../../images/lite/android/tf_execution_flow_android.png)

**Figure 1.** Functional execution flow for TensorFlow Lite models in Android
apps.

At the functional design level, your Android app needs the following elements to
run a TensorFlow Lite model:

-   TensorFlow Lite **runtime environment** for executing the model
-   **Model input handler** to transform data into tensors
-   **Model output handler** to receive output result tensors and interpret them
    as prediction results

The following sections describe how the TensorFlow Lite libraries and tools
provide these functional elements.

## Build apps with TensorFlow Lite

This section describes the recommended, most common path for implementing
TensorFlow Lite in your Android App. You should pay most attention to the
[runtime environment](#runtime) and [development
libraries](#apis) sections. If you have developed a custom
model, make sure to review the [Advanced development
paths](#adv_development) section.

### Runtime environment options {:#runtime}

There are several ways you can enable a runtime environment for executing models
in your Android app. These are the preferred options:

- **TensorFlow Lite in
  [Google Play services runtime environment](./play_services) (recommended)**
- Stand-alone TensorFlow Lite runtime environment

In general, you should use the runtime environment provided by Google Play
services because it is more space-efficient than the standard environment since
it loads dynamically, keeping your app size smaller. Google Play services also
automatically uses the most recent, stable release of the TensorFlow Lite
runtime, giving you additional features and improved performance over time. If
you offer your app on devices that do not include Google Play services or you
need to closely manage your ML runtime environment, then you should use the
standard TensorFlow Lite runtime. This option bundles additional code into your
app, allowing you to have more control over the ML runtime in your app at the
cost of increasing your app's download size.

You access these runtime environments in your Android app by adding TensorFlow
Lite development libraries to your app development environment. For information
about how to use the standard runtime environments in your app, see the next
section.

Note: Some advanced use cases may require customization of model runtime
environment, which are described in the
[Advanced runtime environments](#adv_runtime) section.

### Development APIs and libraries {:#apis}

There are two main APIs you can use to integrate TensorFlow Lite machine
learning models into your Android app:

*   **[TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) (recommended)**
*   [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi)

The
[Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi)
provides classes and methods for running inferences with existing TensorFlow
Lite models. The TensorFlow Lite
[Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary)
wraps the Interpreter API and provides a higher-level programming interface
for performing common machine learning tasks on handling visual, audio, and
text data. You should use the Task API unless you find it does not support
your specific use case.

#### Libraries

You can access the Task APIs or the Interpreter API using the
[Google Play services](./play_services#add_tensorflow_lite_to_your_app).
You can also use the stand-alone libraries for
[TensorFlow Lite Tasks](../inference_with_metadata/task_library/overview) or the
TensorFlow Lite [core](./development#lite_lib) and
[support](./development#support_lib) libraries in your Android app.
For programming details about using TensorFlow Lite libraries and runtime
environments, see [Development tools for Android](./development).

### Obtain models {:#models}

Running a model in an Android app requires a TensorFlow Lite-format model. You
can use prebuilt models or build one with TensorFlow and convert it to the Lite
format. For more information on obtaining models for your Android app, see the
TensorFlow Lite [Models](../models)
section.

### Handle input data {:#input_data}

Any data you pass into a ML model must be a tensor with a specific data
structure, often called the *shape* of the tensor. To process data with a model,
your app code must transform data from its native format, such as image, text,
or audio data, into a tensor in the required shape for your model.

**Note:** Many TensorFlow Lite models come with embedded
[metadata](../inference_with_metadata/overview)
that describes the required input data.

The
[TensorFlow Lite Task library](../inference_with_metadata/task_library/overview)
provides data handling logic for transforming visual, text, and audio data into
tensors with the correct shape to be processed by a TensorFlow Lite model.

### Run inferences {:#inferences}

Processing data through a model to generate a prediction result is known as
running an *inference*. Running an inference in an Android app requires a
TensorFlow Lite [runtime environment](#runtime), a
[model](#models) and [input data](#input_data).

The speed at which a model can generate an inference on a particular device
depends on the size of the data processed, the complexity of the model, and the
available computing resources such as memory and CPU, or specialized processors
called *accelerators*. Machine learning models can run faster on these
specialized processors such as graphics processing units (GPUs) and tensor
processing units (TPUs), using TensorFlow Lite hardware drivers called
*delegates*. For more information about delegates and hardware acceleration of
model processing, see the
[Hardware acceleration overview](../performance/delegates).

### Handle output results {:#output_results}

Models generate prediction results as tensors, which must be handled by your
Android app by taking action or displaying a result to the user. Model output
results can be as simple as a number corresponding to a single result (0 = dog,
1 = cat, 2 = bird) for an image classification, to much more complex results,
such as multiple bounding boxes for several classified objects in an image, with
prediction confidence ratings between 0 and 1.

**Note:** Many TensorFlow Lite models come with embedded
[metadata](../inference_with_metadata/overview)
that describes the output results of a model and how to interpret it.

## Advanced development paths {:#adv_development}

When using more sophisticated and customized TensorFlow Lite models, you may
need to use more advanced development approaches than what is described above.
The following sections describe advanced techniques for executing models and
developing them for TensorFlow Lite in Android apps.

### Advanced runtime environments {:#adv_runtime}

In addition to the standard runtime and Google Play services runtime
environments for TensorFlow Lite, there are additional runtime environments you
can use with your Android app. The most likely use for these environments is if
you have a machine learning model that uses ML operations that are not supported
by the standard runtime environment for TensorFlow Lite.

-   [Flex runtime](../guide/ops_select) for TensorFlow Lite
-   Custom-built TensorFlow Lite runtime

The TensorFlow Lite [Flex runtime](../guide/ops_select) allows you to include
specific operators required for your model. As an advanced option for running
your model, you can build TensorFlow Lite for Android to include operators and
other functionality required for running your TensorFlow machine learning model.
For more information, see [Build TensorFlow Lite for Android](./lite_build).

### C and C++ APIs

TensorFlow Lite also provides an API for running models using C and C++. If your
app uses the [Android NDK](https://developer.android.com/ndk), you should
consider using this API. You may also want to consider using this API if you
want to be able to share code between multiple platforms. For more information
about this development option, see the
[Development tools](./development#tools_for_building_with_c_and_c) page.

### Server-based model execution

In general, you should run models in your app on an Android device to take
advantage of lower latency and improved data privacy for your users. However,
there are cases where running a model on a cloud server, off device, is a better
solution. For example, if you have a large model which does not easily compress
down to a size that fits on your users' Android devices, or can be executed with
reasonable performance on those devices. This approach may also be your
preferred solution if consistent performance of the model across a wide range of
devices is top priority.

Google Cloud offers a full suite of services for running TensorFlow machine
learning models. For more information, see Google Cloud's [AI and machine
learning products](https://cloud.google.com/products/ai) page.

### Custom model development and optimization

More advanced development paths are likely to include developing custom machine
learning models and optimizing those models for use on Android devices. If you
plan to build custom models, make sure you consider applying
[quantization techniques](../performance/post_training_quantization)
to models to reduce memory and processing costs. For more information on how to
build high-performance models for use with TensorFlow Lite, see
[Performance best practices](../performance/best_practices)
in the Models section.

## Next Steps

-   Try the Android [Quickstart](./quickstart) or tutorials
-   Explore the TensorFlow Lite [examples](../examples)
-   Learn how to find or build [TensorFlow Lite models](../models)
