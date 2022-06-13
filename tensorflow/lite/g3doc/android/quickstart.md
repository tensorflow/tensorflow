# Quickstart for Android

This tutorial shows you how to build an Android app using TensorFlow Lite
to analyze a live camera feed and identify objects using a machine learning
model, using a minimal amount of code. If you are updating an existing project
you can use the code sample as a reference and skip ahead to the instructions
for [modifying your project](#add_dependencies).


## Object detection with machine learning

![Object detection animated demo](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/qs-obj-detect.gif){: .attempt-right width="250px"}
The machine learning model in this tutorial performs object detection. An object
detection model takes image data in a specific format, analyzes it, and attempts
to categorize items in the image as one of a set of known classes it was trained
to recognize. The speed at which a model can identify a known object (called an
object *prediction* or *inference*) is usually measured in milliseconds. In
practice, inference speed varies based on the hardware hosting the model, the
size of data being processed, and the size of the machine learning model.


## Setup and run example

For the first part of this tutorial, download the sample from GitHub and run it
using [Android Studio](https://developer.android.com/studio/). The following
sections of this tutorial explore the relevant sections of the code example, so
you can apply them to your own Android apps. You need the following versions
of these tools installed:

* Android Studio 4.2.2 or higher
* Android SDK version 31 or higher

Note: This example uses the camera, so you should runs it on a physical Android
device.

### Get the example code

Create a local copy of the example code. You will use this code to create a
project in Android Studio and run the sample application.

To clone and setup the example code:

1.  Clone the git repository
    <pre class="devsite-click-to-copy">
    git clone https://github.com/android/camera-samples.git
    </pre>
2.  Configure your git instance to use sparse checkout, so you have only
    the files for the object detection example app:
    ```
    cd camera-samples
    git sparse-checkout init --cone
    git sparse-checkout set CameraXAdvanced
    ```

### Import and run the project

Create a project from the downloaded example code, build the project, and then
run it.

To import and build the example code project:

1.  Start [Android Studio](https://developer.android.com/studio).
1.  From the Android Studio **Welcome** page, choose **Import Project**, or
    select **File > New > Import Project**.
1.  Navigate to the example code directory containing the build.gradle file
    (`.../android/camera-samples/CameraXAdvanced/build.gradle`) and select that
    directory.

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
    // from:
    classpath 'com.android.tools.build:gradle:4.2.2'
    // to:
    classpath 'com.android.tools.build:gradle:4.1.2'
    ```
1.  Sync the project by selecting: **File > Sync Project with Gradle Files**.

To run the project:

1.  From Android Studio, run the project by selecting **Run > Run…** and
    **CameraActivity**.
1.  Select an attached Android device with a camera to test the app.

The next sections show you the modifications you need to make to your existing
project to add this functionality to your own app, using this example app as a
reference point.

## Add project dependencies {:#add_dependencies}

In your own application, you must add specific project dependencies to run
TensorFlow Lite machine learning models, and access utility functions that
convert data such as images, into a tensor data format that can be processed by
the model you are using.

The example app uses several TensorFlow Lite libraries to enable execution of
the object detection machine learning model:

-   *TensorFlow Lite main library* - Provides the required data input
    classes,  execution of the machine learning model, and output results from
    the model processing.
-   *TensorFlow Lite Support library* - This library provides a helper class
    to translate images from the camera into a
    [`TensorImage`](../api_docs/java/org/tensorflow/lite/support/image/TensorImage)
    data object that can be processed by the machine learning model.
-   *TensorFlow Lite GPU library* - This library provides support to
    accelerate model execution using GPU processors on the device, if they are
    available.

The following instructions explain how to add the required project and module
dependencies to your own Android app project.

To add module dependencies:

1.  In the module that uses TensorFlow Lite, update the module's
    `build.gradle` file to include the following dependencies. In the example
    code, this file is located here:
    `.../android/camera-samples/CameraXAdvanced/tflite/build.gradle`
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/build.gradle#L69-L71))
    ```
    ...
    dependencies {
    ...
        // Tensorflow lite dependencies
        implementation 'org.tensorflow:tensorflow-lite:2.8.0'
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.8.0'
        implementation 'org.tensorflow:tensorflow-lite-support:2.8.0'
    ...
    }
    ```
1.  In Android Studio, sync the project dependencies by selecting: **File >
    Sync Project with Gradle Files**.

## Initialize the ML model interpreter

In your Android app, you must initialize the TensorFlow Lite machine learning
model interpreter with parameters before running predictions with the model.
These initialization parameters are dependent on the model you are using, and
can include settings such as minimum accuracy thresholds for predictions and
labels for identified object classes.

A TensorFlow Lite model includes a `.tflite` file containing the model code and
frequently includes a labels file containing the names of the classes predicted
by the model. In the case of object detection, classes are objects such as a
person, dog, cat, or car. Models are generally stored in the `src/main/assets`
directory of the primary module, as in the code example:

- CameraXAdvanced/tflite/src/main/assets/coco_ssd_mobilenet_v1_1.0_quant.tflite
- CameraXAdvanced/tflite/src/main/assets/coco_ssd_mobilenet_v1_1.0_labels.txt

For convenience and code readability, the example declares a companion object
that defines the settings for the model.

To initialize the model in your app:

1.  Create a companion object to define the settings for the model:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L342-L347))
    ```
    companion object {
       private val TAG = CameraActivity::class.java.simpleName

       private const val ACCURACY_THRESHOLD = 0.5f
       private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
       private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
    }
    ```
1.  Use the settings from this object to construct a TensorFlow Lite
    [Interpreter](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter)
    object that contains the model:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L90-L94))
    ```
    private val tflite by lazy {
       Interpreter(
           FileUtil.loadMappedFile(this, MODEL_PATH),
           Interpreter.Options().addDelegate(nnApiDelegate))
    }
    ```

### Configure hardware accelerator

When initializing a TensorFlow Lite model in your application, you can
use hardware acceleration features to speed up the prediction
calculations of the model. The code example above uses the NNAPI Delegate to
handle hardware acceleration of the model execution:
```
Interpreter.Options().addDelegate(nnApiDelegate)
```

TensorFlow Lite *delegates* are software modules that accelerate execution of
machine learning models using specialized processing hardware on a mobile
device, such as GPUs, TPUs, or DSPs. Using delegates for running TensorFlow Lite
models is recommended, but not required.

For more information about using delegates with TensorFlow Lite, see
[TensorFlow Lite Delegates](../performance/delegates).


## Provide data to the model

In your Android app, your code provides data to the model for interpretation by
transforming existing data such as images into a
[Tensor](../api_docs/java/org/tensorflow/lite/Tensor)
data format that can be processed by your model. The data in a Tensor must have
specific dimensions, or shape, that matches the format of data used to train the
model.

To determine required tensor shape for a model:

-   Use the initialized
    [Interpreter](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter)
    object to determine the shape of the tensor used by your model, as shown in
    the code snippet below:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L102-L106))
    ```
    private val tfInputSize by lazy {
       val inputIndex = 0
       val inputShape = tflite.getInputTensor(inputIndex).shape()
       Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }
    ```

The object detection model used in the example code expects square images with a
size of 300 by 300 pixels.

Before you can provide images from the camera, your app must take the image,
make it conform to the expected size, adjust its rotation, and normalize the
image data. When processing images with a TensorFlow Lite model, you can use the
TensorFlow Lite Support Library
[ImageProcessor](../api_docs/java/org/tensorflow/lite/support/image/ImageProcessor)
class to handle this data pre-processing, as show below.

To transform image data for a model:

1.  Use the Support Library
    [ImageProcessor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/ImageProcessor)
    to create an object for transforming image data into a format that your
    model can use to run predictions:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L75-L84))
    ```
    private val tfImageProcessor by lazy {
       val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
       ImageProcessor.Builder()
           .add(ResizeWithCropOrPadOp(cropSize, cropSize))
           .add(ResizeOp(
               tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
           .add(Rot90Op(-imageRotationDegrees / 90))
           .add(NormalizeOp(0f, 1f))
           .build()
    }
    ```
1.  Copy the image data from the Android camera system and prepare it for
    analysis with your
    [ImageProcessor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/ImageProcessor)
    object:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L198-L202))
    ```
    // Copy out RGB bits to the shared buffer
    image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)  }

    // Process the image in Tensorflow
    val tfImage =  tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
    ```

Note: When extracting image information from the Android camera subsystem, make
sure to get an image in RGB format. This format is required by the TensorFlow
Lite
[ImageProcessor](../api_docs/java/org/tensorflow/lite/support/image/ImageProcessor)
class which you use to prepare the image for analysis by a model. If the
RGB-format image contains an Alpha channel, that transparency data is ignored.


## Run predictions

In your Android app, once you create a
[TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage)
object with image data in the correct format, you can run the model against that
data to produce a prediction, or *inference*. The example code for this tutorial
uses an
[ObjectDetectionHelper](https://github.com/android/camera-samples/blob/main/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt)
class that encapsulates this code in a `predict()` method.

To run a prediction on a set of image data:

1.  Run the prediction by passing the image data to your predict function:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L204-L205))
    ```
    // Perform the object detection for the current frame
    val predictions = detector.predict(tfImage)
    ```
1.  Call the run method on your `tflite` object instance with
    the image data to generate predictions:
    ([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt#L60-L63))
    ```
    fun predict(image: TensorImage): List<ObjectPrediction> {
       tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)
       return predictions
    }
    ```

The TensorFlow Lite Interpreter object receives this data, runs it against the
model, and produces a list of predictions. For continuous processing of data by
the model, use the `runForMultipleInputsOutputs()` method so that Interpreter
objects are not created and then removed by the system for each prediction run.

## Handle model output

In your Android app, after you run image data against the object detection
model, it produces a list of predictions which your app code must handle
by executing additional business logic, displaying results to the user, or
taking other actions.

The output of any given TensorFlow Lite model varies in terms of the number of
predictions it produces (one or many), and the descriptive information for each
prediction. In the case of an object detection model, predictions typically
include data for a bounding box that indicates where an object is detected in
the image. In the example code, the returned data is formatted as a list of
[ObjectPrediction](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt#L42-L58)
objects, as shown below:
([code reference](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt#L42-L58))

```
val predictions get() = (0 until OBJECT_COUNT).map {
   ObjectPrediction(

       // The locations are an array of [0, 1] floats for [top, left, bottom, right]
       location = locations[0][it].let {
           RectF(it[1], it[0], it[3], it[2])
       },

       // SSD Mobilenet V1 Model assumes class 0 is background class
       // in label file and class labels start from 1 to number_of_classes + 1,
       // while outputClasses correspond to class index from 0 to number_of_classes
       label = labels[1 + labelIndices[0][it].toInt()],

       // Score is a single value of [0, 1]
       score = scores[0][it]
   )
}
```

![Object detection screenshot](../../images/lite/android/qs-obj-detect.jpeg){: .attempt-right width="250px"}
For the model used in this example, each prediction includes a bounding box
location for the object, a label for the object, and a prediction score between
0 and 1 as a Float representing the confidence of the prediction, with 1 being
the highest confidence rating. In general, predictions with a score below 50%
(0.5) are considered inconclusive. However, how you handle low-value prediction
results is up to you and the needs of your application.

Once the model has returned a prediction result, your application can act on
that prediction by presenting the result to your user or executing additional
logic. In the case of the example code, the application draws a bounding box
around the identified object and displays the class name on screen. Review the
[`CameraActivity.reportPrediction()`](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L236-L262)
function in the example code for details.

## Next steps

*   Explore various uses of TensorFlow Lite in the [examples](../examples).
*   Learn more about using machine learning models with TensorFlow Lite
    in the [Models](../models) section.
*   Learn more about implementing machine learning in your mobile
    application in the [TensorFlow Lite Developer Guide](../guide).
