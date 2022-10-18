# Object detection with Android

This tutorial shows you how to build an Android app using TensorFlow Lite to
continuously detect objects in frames captured by a device camera. This
application is designed for a physical Android device. If you are updating an
existing project, you can use the code sample as a reference and skip ahead to
the instructions for [modifying your project](#add_dependencies).

![Object detection animated demo](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}
## Object detection overview

*Object detection* is the machine learning task of identifying the presence and
location of multiple classes of objects within an image. An object detection
model is trained on a dataset that contains a set of known objects.

The trained model receives image frames as input and attempts to categorize
items in the images from the set of known classes it was trained to recognize.
For each image frame, the object detection model outputs a list of the objects
it detects, the location of a bounding box for each object, and a score that
indicates the confidence of the object being correctly classified.

## Models and dataset

This tutorial uses models that were trained using the
[COCO dataset](http://cocodataset.org/). COCO is a large-scale object detection
dataset that contains 330K images, 1.5 million object instances, and 80 object
categories.

You have the option to use one of the following pre-trained models:

*   [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1)
    *[Recommended]* - a lightweight object detection model with a BiFPN feature
    extractor, shared box predictor, and focal loss. The mAP (mean Average
    Precision) for the COCO 2017 validation dataset is 25.69%.

*   [EfficientDet-Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1) -
    a medium-sized EfficientDet object detection model. The mAP for the COCO
    2017 validation dataset is 30.55%.

*   [EfficientDet-Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1) -
    a larger EfficientDet object detection model. The mAP for the COCO 2017
    validation dataset is 33.97%.

*   [MobileNetV1-SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2) -
    an extremely lightweight model optimized to work with TensorFlow Lite for
    object detection. The mAP for the COCO 2017 validation dataset is 21%.

For this tutorial, the *EfficientDet-Lite0* model strikes a good balance between
size and accuracy.

Downloading, extraction, and placing the models into the assets folder is
managed automatically by the `download.gradle` file, which is run at build time.
You don't need to manually download TFLite models into the project.

## Setup and run example

To setup the object detection app, download the sample from
[GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)
and run it using [Android Studio](https://developer.android.com/studio/). The
following sections of this tutorial explore the relevant sections of the code
example, so you can apply them to your own Android apps.

### System requirements

*   **[Android Studio](https://developer.android.com/studio/index.html)**
    version 2021.1.1 (Bumblebee) or higher.
*   Android SDK version 31 or higher
*   Android device with a minimum OS version of SDK 24 (Android 7.0 -
    Nougat) with developer mode enabled.

Note: This example uses a camera, so run it on a physical Android device.

### Get the example code

Create a local copy of the example code. You will use this code to create a
project in Android Studio and run the sample application.

To clone and setup the example code:

1.  Clone the git repository
    <pre class="devsite-click-to-copy">
    git clone https://github.com/tensorflow/examples.git
    </pre>
2.  Optionally, configure your git instance to use sparse checkout, so you have only
    the files for the object detection example app:
    <pre class="devsite-click-to-copy">
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/object_detection/android
    </pre>

### Import and run the project

Create a project from the downloaded example code, build the project, and then
run it.

To import and build the example code project:

1.  Start [Android Studio](https://developer.android.com/studio).
1.  From the Android Studio, select **File > New > Import Project**.
1.  Navigate to the example code directory containing the build.gradle file
    (`.../examples/lite/examples/object_detection/android/build.gradle`) and
    select that directory.
1.  If Android Studio requests a Gradle Sync, choose OK.
1.  Ensure that your Android device is connected to your computer and developer
    mode is enabled. Click the green `Run` arrow.

If you select the correct directory, Android Studio creates a new project and
builds it. This process can take a few minutes, depending on the speed of your
computer and if you have used Android Studio for other projects. When the build
completes, the Android Studio displays a `BUILD SUCCESSFUL` message in the
**Build Output** status panel.

Note: The example code is built with Android Studio 4.2.2, but works with
earlier versions of Studio. If you are using an earlier version of Android
Studio you can try adjust the version number of the Android plugin so that the
build completes, instead of upgrading Studio.

**Optional:** To fix build errors by updating the Android plugin version:

1.  Open the build.gradle file in the project directory.
1.  Change the Android tools version as follows:

    ```
    // from: classpath
    'com.android.tools.build:gradle:4.2.2'
    // to: classpath
    'com.android.tools.build:gradle:4.1.2'
    ```

1.  Sync the project by selecting: **File > Sync Project with Gradle Files**.

To run the project:

1.  From Android Studio, run the project by selecting **Run > Run…**.
1.  Select an attached Android device with a camera to test the app.

The next sections show you the modifications you need to make to your existing
project to add this functionality to your own app, using this example app as a
reference point.

## Add project dependencies {:#add_dependencies}

In your own application, you must add specific project dependencies to run
TensorFlow Lite machine learning models, and access utility functions that
convert data such as images, into a tensor data format that can be processed by
the model you are using.

The example app uses the TensorFlow Lite
[Task library for vision](../../inference_with_metadata/task_library/overview#supported_tasks)
to enable execution of the object detection machine learning model. The
following instructions explain how to add the required library dependencies to
your own Android app project.

The following instructions explain how to add the required project and module
dependencies to your own Android app project.

To add module dependencies:

1.  In the module that uses TensorFlow Lite, update the module's `build.gradle`
    file to include the following dependencies. In the example code, this file
    is located here:
    `...examples/lite/examples/object_detection/android/app/build.gradle`
    ([code reference](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/build.gradle))

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    The project must include the Vision task library
    (`tensorflow-lite-task-vision`). The graphics processing unit (GPU) library
    (`tensorflow-lite-gpu-delegate-plugin`) provides the infrastructure to run
    the app on GPU, and Delegate (`tensorflow-lite-gpu`) provides the
    compatibility list.

1.  In Android Studio, sync the project dependencies by selecting: **File > Sync
    Project with Gradle Files**.

## Initialize the ML model

In your Android app, you must initialize the TensorFlow Lite machine learning
model with parameters before running predictions with the model. These
initialization parameters are consistent across object detection models and can
include settings such as minimum accuracy thresholds for predictions.

A TensorFlow Lite model includes a `.tflite` file containing the model code and
frequently includes a labels file containing the names of the classes predicted
by the model. In the case of object detection, classes are objects such as a
person, dog, cat, or car.

This example downloads several models that are specified in
`download_models.gradle`, and the `ObjectDetectorHelper` class provides a
selector for the models:

```
val modelName =
  when (currentModel) {
    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
    else -> "mobilenetv1.tflite"
  }
```

Key Point: Models should be stored in the `src/main/assets` directory of your
development project. The TensorFlow Lite Task library automatically checks this
directory when you specify a model file name.

To initialize the model in your app:

1.  Add a `.tflite` model file to the `src/main/assets` directory of your
    development project, such as:
    [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1).
1.  Set a static variable for your model's file name. In the example app, you
    set the `modelName` variable to `MODEL_EFFICIENTDETV0` to use the
    EfficientDet-Lite0 detection model.
1.  Set the options for model, such as the prediction threshold, results set
    size, and optionally, hardware acceleration delegates:

    ```
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(threshold)
        .setMaxResults(maxResults)
    ```

1.  Use the settings from this object to construct a TensorFlow Lite
    [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile\(Context,%20java.lang.String\))
    object that contains the model:

    ```
    objectDetector =
      ObjectDetector.createFromFileAndOptions(
        context, modelName, optionsBuilder.build())
    ```

The `setupObjectDetector` sets up the following model parameters:

*   Detection threshold
*   Maximum number of detection results
*   Number of processing threads to use
    (`BaseOptions.builder().setNumThreads(numThreads)`)
*   Actual model (`modelName`)
*   ObjectDetector object (`objectDetector`)

### Configure hardware accelerator

When initializing a TensorFlow Lite model in your application, you can use
hardware acceleration features to speed up the prediction calculations of the
model.

TensorFlow Lite *delegates* are software modules that accelerate execution of
machine learning models using specialized processing hardware on a mobile
device, such as Graphics Processing Units (GPUs), Tensor Processing Units
(TPUs), and Digital Signal Processors (DSPs). Using delegates for running
TensorFlow Lite models is recommended, but not required.

The object detector is initialized using the current settings on the thread that
is using it. You can use CPU and [NNAPI](../../android/delegates/nnapi)
delegates with detectors that are created on the main thread and used on a
background thread, but the thread that initialized the detector must use the GPU
delegate.

The delegates are set within the `ObjectDetectionHelper.setupObjectDetector()`
function:

```
when (currentDelegate) {
    DELEGATE_CPU -> {
        // Default
    }
    DELEGATE_GPU -> {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else {
            objectDetectorListener?.onError("GPU is not supported on this device")
        }
    }
    DELEGATE_NNAPI -> {
        baseOptionsBuilder.useNnapi()
    }
}
```

For more information about using hardware acceleration delegates with TensorFlow
Lite, see [TensorFlow Lite Delegates](../../performance/delegates).

## Prepare data for the model

In your Android app, your code provides data to the model for interpretation by
transforming existing data such as image frames into a Tensor data format that
can be processed by your model. The data in a Tensor you pass to a model must
have specific dimensions, or shape, that matches the format of data used to
train the model.

The
[EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1)
model used in this code example accepts Tensors representing images with a
dimension of 320 x 320, with three channels (red, blue, and green) per pixel.
Each value in the tensor is a single byte between 0 and 255. So, to run
predictions on new images, your app must transform that image data into Tensor
data objects of that size and shape. The TensorFlow Lite Task Library Vision API
handles the data transformation for you.

The app uses an
[`ImageAnalysis`](https://developer.android.com/training/camerax/analyze) object
to pull images from the camera. This object calls the `detectObject` function
with bitmap from the camera. The data is automatically resized and rotated by
`ImageProcessor` so that it meets the image data requirements of the model. The
image is then translated into a
[`TensorImage`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/TensorImage)
object.

To prepare data from the camera subsystem to be processed by the ML model:

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

1.  Connect the analyzer to the camera subsystem and create a bitmap buffer to
    contain the data received from the camera:

    ```
    .also {
      it.setAnalyzer(cameraExecutor) {
        image -> if (!::bitmapBuffer.isInitialized)
        { bitmapBuffer = Bitmap.createBitmap( image.width, image.height,
        Bitmap.Config.ARGB_8888 ) } detectObjects(image)
        }
      }
    ```

1.  Extract the specific image data needed by the model, and pass the image
    rotation information:

    ```
    private fun detectObjects(image: ImageProxy) {
      //Copy out RGB bits to the shared bitmap buffer
      image.use {bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
      }
    ```

1.  Complete any final data transformations and add the image data to a
    `TensorImage` object, as shown in the `ObjectDetectorHelper.detect()` method
    of the example app:

    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

Note: When extracting image information from the Android camera subsystem, make
sure to get an image in RGB format. This format is required by the TensorFlow
Lite ImageProcessor class which you use to prepare the image for analysis by a
model. If the RGB-format image contains an Alpha channel, that transparency data
is ignored.

## Run predictions

In your Android app, once you create a TensorImage object with image data in the
correct format, you can run the model against that data to produce a prediction,
or *inference*.

In the `fragments/CameraFragment.kt` class of the example app, the
`imageAnalyzer` object within the `bindCameraUseCases` function automatically
passes data to the model for predictions when the app is connected to the
camera.

The app uses the `cameraProvider.bindToLifecycle()` method to handle the camera
selector, preview window, and ML model processing. The `ObjectDetectorHelper.kt`
class handles passing the image data into the model. To run the model and
generate predictions from image data:

-   Run the prediction by passing the image data to your predict function:

    ```
    val results = objectDetector?.detect(tensorImage)
    ```

The TensorFlow Lite Interpreter object receives this data, runs it against the
model, and produces a list of predictions. For continuous processing of data by
the model, use the `runForMultipleInputsOutputs()` method so that Interpreter
objects are not created and then removed by the system for each prediction run.

## Handle model output

In your Android app, after you run image data against the object detection
model, it produces a list of predictions which your app code must handle by
executing additional business logic, displaying results to the user, or taking
other actions.

The output of any given TensorFlow Lite model varies in terms of the number of
predictions it produces (one or many), and the descriptive information for each
prediction. In the case of an object detection model, predictions typically
include data for a bounding box that indicates where an object is detected in
the image. In the example code, the results are passed to the `onResults`
function in `CameraFragment.kt`, which is acting as a DetectorListener on the
object detection process.

```
interface DetectorListener {
  fun onError(error: String)
  fun onResults(
    results: MutableList<Detection>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int
  )
}
```

For the model used in this example, each prediction includes a bounding box
location for the object, a label for the object, and a prediction score between
0 and 1 as a Float representing the confidence of the prediction, with 1 being
the highest confidence rating. In general, predictions with a score below 50%
(0.5) are considered inconclusive. However, how you handle low-value prediction
results is up to you and the needs of your application.

To handle model prediction results:

1.  Use a listener pattern to pass results to your app code or user interface
    objects. The example app uses this pattern to pass detection results from
    the `ObjectDetectorHelper` object to the `CameraFragment` object:

    ```
    objectDetectorListener.onResults(
    // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```

1.  Act on the results, such as displaying the prediction to the user. The
    example draws an overlay on the CameraPreview object to show the result:

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

Once the model has returned a prediction result, your application can act on
that prediction by presenting the result to your user or executing additional
logic. In the case of the example code, the application draws a bounding box
around the identified object and displays the class name on screen.

## Next steps

*   Explore various uses of TensorFlow Lite in the [examples](../../examples).
*   Learn more about using machine learning models with TensorFlow Lite in the
    [Models](../../models) section.
*   Learn more about implementing machine learning in your mobile application in
    the [TensorFlow Lite Developer Guide](../../guide).
