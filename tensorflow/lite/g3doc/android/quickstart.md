# Quickstart for Android

This page shows you how to build an Android app with TensorFlow Lite to analyze
a live camera feed and identify objects. This machine learning use case is
called *object detection*. The example app uses the TensorFlow Lite
[Task library for vision](../inference_with_metadata/task_library/overview#supported_tasks)
via [Google Play services](./play_services) to enable execution of the object
detection machine learning model, which is the recommended approach for building
an ML application with TensorFlow Lite.

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in
Google Play services APIs, you agree to the <a href="./play_services#tos">Terms
of Service</a>. Please read and understand all applicable terms and policies
before accessing the APIs. </aside>

![Object detection animated demo](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}
## Setup and run the example

For the first part of this exercise, download the
[example code](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services)
from GitHub and run it using [Android Studio](https://developer.android.com/studio/).
The following sections of this document explore the relevant sections of the
code example, so you can apply them to your own Android apps. You need the
following versions of these tools installed:

* Android Studio 4.2 or higher
* Android SDK version 21 or higher

Note: This example uses the camera, so you should run it on a physical Android
device.

### Get the example code

Create a local copy of the example code so you can build and run it.

To clone and setup the example code:

1.  Clone the git repository
    <pre class="devsite-click-to-copy">
    git clone https://github.com/tensorflow/examples.git
    </pre>
2.  Configure your git instance to use sparse checkout, so you have only
    the files for the object detection example app:
    <pre class="devsite-click-to-copy">
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/object_detection/android_play_services
    </pre>

### Import and run the project

Use Android Studio to create a project from the downloaded example code, build
the project, and run it.

To import and build the example code project:

1.  Start [Android Studio](https://developer.android.com/studio).
1.  From the Android Studio **Welcome** page, choose **Import Project**, or
    select **File > New > Import Project**.
1.  Navigate to the example code directory containing the build.gradle file
    (`...examples/lite/examples/object_detection/android_play_services/build.gradle`)
    and select that directory.

After you select this directory, Android Studio creates a new project and builds
it. When the build completes, the Android Studio displays a `BUILD SUCCESSFUL`
message in the **Build Output** status panel.

To run the project:

1.  From Android Studio, run the project by selecting **Run > Run…** and
    **MainActivity**.
1.  Select an attached Android device with a camera to test the app.

## How the example app works

The example app uses pre-trained object detection model, such as
[mobilenetv1.tflite](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite),
in TensorFlow Lite format look for objects in a live video stream from an
Android device's camera. The code for this feature is primarily in these files:

*   [ObjectDetectorHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper.kt) -
    Initializes the runtime environment, enables hardware acceleration, and 
    runs the object detection ML model.
*   [CameraFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/fragments/CameraFragment.kt) -
    Builds the camera image data stream, prepares data for the model, and
    displays the object detection results.

Note: This example app uses the TensorFlow Lite
[Task Library](../inference_with_metadata/task_library/overview#supported_tasks),
which provides easy-to-use, task-specific APIs for performing common machine
learning operations. For apps with more specific needs and customized ML
functions, consider using the
[Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).

The next sections show you the key components of these code files, so you can
modify an Android app to add this functionality.

## Build the app {:#build_app}

The following sections explain the key steps to build your own Android app and
run the model shown in the example app. These instructions use the example app
shown earlier as a reference point.

Note: To follow along with these instructions and build your own app, create a
[basic Android project](https://developer.android.com/studio/projects/create-project)
using Android Studio.

### Add project dependencies {:#add_dependencies}

In your basic Android app, add the project dependencies for running TensorFlow
Lite machine learning models and accessing ML data utility functions. These
utility functions convert data such as images into a tensor data format that can
be processed by a model.

The example app uses the TensorFlow Lite
[Task library for vision](../inference_with_metadata/task_library/overview#supported_tasks)
from [Google Play services](./play_services) to enable execution of the object
detection machine learning model. The following instructions explain how to add
the required library dependencies to your own Android app project.

To add module dependencies:

1.  In the Android app module that uses TensorFlow Lite, update the module's
    `build.gradle` file to include the following dependencies. In the example
    code, this file is located here:
    `...examples/lite/examples/object_detection/android_play_services/app/build.gradle`
    ```
    ...
    dependencies {
    ...
        // Tensorflow Lite dependencies
        implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
        implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ...
    }
    ```
1.  In Android Studio, sync the project dependencies by selecting: **File >
    Sync Project with Gradle Files**.

### Initialize Google Play services

When you use [Google Play services](./play_services) to run TensorFlow Lite
models, you must initialize the service before you can use it. If you want to
use hardware acceleration support with the service, such as GPU acceleration,
you also enable that support as part of this initialization.

To initialize TensorFlow Lite with Google Play services:

1.  Create a `TfLiteInitializationOptions` object and modify it to enable GPU
    support:

    ```
    val options = TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ```

1.  Use the `TfLiteVision.initialize()` method to enable use of the Play
    services runtime, and set a listener to verify that it loaded successfully:

    ```
    TfLiteVision.initialize(context, options).addOnSuccessListener {
        objectDetectorListener.onInitialized()
    }.addOnFailureListener {
        // Called if the GPU Delegate is not supported on the device
        TfLiteVision.initialize(context).addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener{
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }
    ```

### Initialize the ML model interpreter

Initialize the TensorFlow Lite machine learning model interpreter by loading the
model file and setting model parameters. A TensorFlow Lite model includes a
`.tflite` file containing the model code. You should store your models in the
`src/main/assets` directory of your development project, for example:

```
.../src/main/assets/mobilenetv1.tflite`
```

Tip: Task library interpreter code automatically looks for models in the
`src/main/assets` directory if you do not specify a file path.

To initialize the model:

1.  Add a `.tflite` model file to the `src/main/assets` directory of your
    development project, such as [ssd_mobilenet_v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2).
1.  Set the `modelName` variable to specify your ML model's file name:
    ```
    val modelName = "mobilenetv1.tflite"
    ```
1.  Set the options for model, such as the prediction threshold and results set
    size:
    ```
    val optionsBuilder =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
    ```
1.  Enable GPU acceleration with the options and allow the code to fail
    gracefully if acceleration is not supported on the device:
    ```
    try {
        optionsBuilder.useGpu()
    } catch(e: Exception) {
        objectDetectorListener.onError("GPU is not supported on this device")
    }

    ```
1.  Use the settings from this object to construct a TensorFlow Lite
    [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String))
    object that contains the model:
    ```
    objectDetector =
        ObjectDetector.createFromFileAndOptions(
            context, modelName, optionsBuilder.build())
    ```

For more information about using hardware acceleration delegates with TensorFlow
Lite, see [TensorFlow Lite Delegates](../performance/delegates).

### Prepare data for the model

You prepare data for interpretation by the model by transforming existing data
such as images into the [Tensor](../api_docs/java/org/tensorflow/lite/Tensor)
data format, so it can be processed by your model. The data in a Tensor must
have specific dimensions, or shape, that matches the format of data used to
train the model. Depending on the model you use, you may need to transform the
data to fit what the model expects. The example app uses an
[`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis)
object to extract image frames from the camera subsystem.

To prepare data for processing by the model:

1.  Build an `ImageAnalysis` object to extract images in the required format:
    ```
    imageAnalyzer =
        ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            ...
    ```
1.  Connect the analyzer to the camera subsystem and create a bitmap buffer
    to contain the data received from the camera:
    ```
            .also {
            it.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width,
                        image.height,
                        Bitmap.Config.ARGB_8888
                    )
                }
                detectObjects(image)
            }
        }
    ```
1.  Extract the specific image data needed by the model, and pass
    the image rotation information:
    ```
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }    
    ```
1.  Complete any final data transformations and add the image data to a
    `TensorImage` object, as shown in the `ObjectDetectorHelper.detect()`
    method of the example app:
    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

### Run predictions

Once you create a
[TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage)
object with image data in the correct format, you can run the model against that
data to produce a prediction, or *inference*. In the example app, this code
is contained in the `ObjectDetectorHelper.detect()` method.

To run a the model and generate predictions from image data:

-   Run the prediction by passing the image data to your predict function:
    ```
    val results = objectDetector?.detect(tensorImage)
    ```

### Handle model output

After you run image data against the object detection model, it produces a list
of prediction results which your app code must handle by executing additional
business logic, displaying results to the user, or taking other actions. The
object detection model in the example app produces a list of predictions and
bounding boxes for the detected objects. In the example app, the prediction
results are passed to a listener object for further processing and display to
the user.

To handle model prediction results:

1.  Use a listener pattern to pass results to your app code or user interface
    objects. The example app uses this pattern to pass detection results from
    the `ObjectDetectorHelper` object to the `CameraFragment` object:
    ```
    objectDetectorListener.onResults( // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```
1.  Act on the results, such as displaying the prediction to the user. The 
    example app draws an overlay on the `CameraPreview` object to show the result:
    ```
    override fun onResults(
      results: MutableList<Detection>?,
      inferenceTime: Long,
      imageHeight: Int,
      imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }
    ```

## Next steps

*   Learn more about the
    [Task Library APIs](../inference_with_metadata/task_library/overview#supported_tasks)
*   Learn more about the
    [Interpreter APIs](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).
*   Explore the uses of TensorFlow Lite in the [examples](../examples).
*   Learn more about using and building machine learning models with TensorFlow
    Lite in the [Models](../models) section.
*   Learn more about implementing machine learning in your mobile application in
    the [TensorFlow Lite Developer Guide](../guide).
