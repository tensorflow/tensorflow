# Sound and word recognition for Android

This tutorial shows you how to use TensorFlow Lite with pre-built machine
learning models to recognize sounds and spoken words in an Android app. Audio
classification models like the ones shown in this tutorial can be used to detect
activity, identify actions, or recognize voice commands.

This tutorial shows you how to download the example code, load the project into
[Android Studio](https://developer.android.com/studio/), and explains key parts
of the code example so you can start adding this functionality to your own app.
The example app code uses the TensorFlow
[Task Library for Audio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier),
which handles most of the audio data recording and preprocessing. For more
information on how audio is pre-processed for use with machine learning models,
see
[Audio Data Preparation and Augmentation](https://www.tensorflow.org/io/tutorials/audio).

## Audio classification with machine learning

The machine learning model in this tutorial recognizes sounds or words from
audio samples recorded with a microphone on an Android device. The example app
in this tutorial allows you to switch between the
[YAMNet/classifier](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1),
a model that recognizes sounds, and a model that recognizes specific spoken
words, that was
[trained](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition)
using the TensorFlow Lite [Model
Maker](https://www.tensorflow.org/lite/models/modify/model_maker) tool. The
models run predictions on audio clips that contain 15600 individual samples per
clip and are about 1 second in length.

## Setup and run example

For the first part of this tutorial, you download the sample from GitHub and run
it using Android Studio. The following sections of this tutorial explore the
relevant sections of the example, so you can apply them to your own Android
apps.

### System requirements

*   [Android Studio](https://developer.android.com/studio/index.html)
    version 2021.1.1 (Bumblebee) or higher.
*   Android SDK version 31 or higher
*   Android device with a minimum OS version of SDK 24 (Android 7.0 -
    Nougat) with developer mode enabled.

### Get the example code

Create a local copy of the example code. You will use this code to create a
project in Android Studio and run the sample application.

To clone and setup the example code:
1.  Clone the git repository
    <pre class="devsite-click-to-copy">
    git clone https://github.com/tensorflow/examples.git
    </pre>
1.  Optionally, configure your git instance to use sparse checkout, so you
    have only the files for the example app:
    ```
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/audio_classification/android
    ```

### Import and run the project

Create a project from the downloaded example code, build the project, and then
run it.

To import and build the example code project:
1.  Start [Android Studio](https://developer.android.com/studio).
1.  In Android Studio, choose **File > New > Import Project**.
1.  Navigate to the example code directory containing the `build.gradle` file
    (`.../examples/lite/examples/audio_classification/android/build.gradle`)
    and select that directory.

If you select the correct directory, Android Studio creates a new project and
builds it. This process can take a few minutes, depending on the speed of your
computer and if you have used Android Studio for other projects. When the build
completes, the Android Studio displays a `BUILD SUCCESSFUL` message in the
**Build Output** status panel.

To run the project:
1.  From Android Studio, run the project by selecting **Run > Run 'app'**.
1.  Select an attached Android device with a microphone to test the app.

Note: If you use an emulator to run the app, make sure you
[enable audio input](https://developer.android.com/studio/releases/emulator#29.0.6-host-audio)
from the host machine.

The next sections show you the modifications you need to make to your existing
project to add this functionality to your own app, using this example app as a
reference point.

## Add project dependencies

In your own application, you must add specific project dependencies to run
TensorFlow Lite machine learning models, and access utility functions that
convert standard data formats, such as audio, into a tensor data format that can
be processed by the model you are using.

The example app uses the following TensorFlow Lite libraries:

-   [TensorFlow Lite Task library Audio API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/package-summary) -
    Provides the required audio data input classes, execution of the machine
    learning model, and output results from the model processing.

The following instructions show how to add the required project dependencies to
your own Android app project.

To add module dependencies:
1.  In the module that uses TensorFlow Lite, update the module's
    `build.gradle` file to include the following dependencies. In the example
    code, this file is located here:
    `.../examples/lite/examples/audio_classification/android/build.gradle`
    ```
    dependencies {
    ...
        implementation 'org.tensorflow:tensorflow-lite-task-audio'
    }
    ```
1.  In Android Studio, sync the project dependencies by selecting: **File >
    Sync Project with Gradle Files**.

## Initialize the ML model

In your Android app, you must initialize the TensorFlow Lite machine learning
model with parameters before running predictions with the model. These
initialization parameters are dependent on the model and can include settings
such as default minimum accuracy thresholds for predictions and labels for words
or sounds that the model can recognize.

A TensorFlow Lite model includes a `*.tflite` file containing the model. The
model file contains the prediction logic and typically includes [metadata](../../models/convert/metadata)
about how to interpret prediction results, such as prediction class names. Model
files should be stored in the `src/main/assets` directory of your development
project, as in the code example:

-   `<project>/src/main/assets/yamnet.tflite`

For convenience and code readability, the example declares a companion object
that defines the settings for the model.

To initialize the model in your app:

1.  Create a companion object to define the settings for the model:
    ```
    companion object {
      const val DISPLAY_THRESHOLD = 0.3f
      const val DEFAULT_NUM_OF_RESULTS = 2
      const val DEFAULT_OVERLAP_VALUE = 0.5f
      const val YAMNET_MODEL = "yamnet.tflite"
      const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
    ```
1.  Create the settings for the model by building an
    `AudioClassifier.AudioClassifierOptions` object:
    ```
    val options = AudioClassifier.AudioClassifierOptions.builder()
      .setScoreThreshold(classificationThreshold)
      .setMaxResults(numOfResults)
      .setBaseOptions(baseOptionsBuilder.build())
      .build()
    ```
1.  Use this settings object to construct a TensorFlow Lite
    [`AudioClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)
    object that contains the model:
    ```
    classifier = AudioClassifier.createFromFileAndOptions(context, "yamnet.tflite", options)
    ```

### Enable hardware acceleration

When initializing a TensorFlow Lite model in your app, you should consider using
hardware acceleration features to speed up the prediction calculations of the
model. TensorFlow Lite
[delegates](https://www.tensorflow.org/lite/performance/delegates) are software
modules that accelerate execution of machine learning models using specialized
processing hardware on a mobile device, such as graphics processing unit (GPUs)
or tensor processing units (TPUs). The code example uses the NNAPI Delegate to
handle hardware acceleration of the model execution:

```
val baseOptionsBuilder = BaseOptions.builder()
   .setNumThreads(numThreads)
...
when (currentDelegate) {
   DELEGATE_CPU -> {
       // Default
   }
   DELEGATE_NNAPI -> {
       baseOptionsBuilder.useNnapi()
   }
}
```

Using delegates for running TensorFlow Lite models is recommended, but not
required. For more information about using delegates with TensorFlow Lite, see
[TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates).

## Prepare data for the model

In your Android app, your code provides data to the model for interpretation by
transforming existing data such as audio clips into a
[Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor)
data format that can be processed by your model. The data in a Tensor you pass
to a model must have specific dimensions, or shape, that matches the format of
data used to train the model.

The
[YAMNet/classifier model](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)
and the customized
[speech commands](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition)
models used in this code example accepts Tensor data objects that represent
single-channel, or mono, audio clips recorded at 16kHz in 0.975 second clips
(15600 samples). Running predictions on new audio data, your app must transform
that audio data into Tensor data objects of that size and shape. The TensorFlow
Lite Task Library
[Audio API](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)
handles the data transformation for you.

In the example code `AudioClassificationHelper` class, the app records live
audio from the device microphones using an Android
[AudioRecord](https://developer.android.com/reference/android/media/AudioRecord)
object. The code uses
[AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier)
to build and configure that object to record audio at a sampling rate
appropriate for the model. The code also uses AudioClassifier to build a
[TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio)
object to store the transformed audio data. Then the TensorAudio object is
passed to the model for analysis.

To provide audio data to the ML model:
-   Use the `AudioClassifier` object to create a `TensorAudio` object and a
    `AudioRecord` object:
    ```
    fun initClassifier() {
    ...
      try {
        classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
        // create audio input objects
        tensorAudio = classifier.createInputTensorAudio()
        recorder = classifier.createAudioRecord()
      }
    ```

Note: Your app must request permission to record audio using an Android device
microphone. See the `fragments/PermissionsFragment` class in the project for an
example. For more information on requesting permissions, see
[Permissions on Android](https://developer.android.com/guide/topics/permissions/overview).

## Run predictions

In your Android app, once you have connected an
[AudioRecord](https://developer.android.com/reference/android/media/AudioRecord)
object and a
[TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio)
object to an AudioClassifier object,  you can run the model against that data to
produce a prediction, or *inference*. The example code for this tutorial runs
predictions on clips from a live-recorded audio input stream at a specific
rate.

Model execution consumes significant resources, so it's important to run ML
model predictions on a separate, background thread. The example app uses a
`[ScheduledThreadPoolExecutor](https://developer.android.com/reference/java/util/concurrent/ScheduledThreadPoolExecutor)`
object to isolate the model processing from other functions of the app.

Audio classification models that recognize sounds with a clear beginning and
end, such as words, can produce more accurate predictions on an incoming audio
stream by analyzing overlapping audio clips. This approach
helps the model avoid missing predictions for words that are cut off at the end
of a clip. In the example app, each time you run a prediction the code grabs the
latest 0.975 second clip from the audio recording buffer and analyzes it. You
can make the model analyze overlapping audio clips by setting the model analysis
thread execution pool `interval` value to a length that's shorter than the
length of the clips being analyzed. For example, if your model analyzes 1 second
clips and you set the interval to 500 milliseconds, the model will analyze the
last half of the previous clip and 500 milliseconds of new audio data each time,
creating a clip analysis overlap of 50%.

To start running predictions on the audio data:

1.  Use the `AudioClassificationHelper.startAudioClassification()` method
    to start the audio recording for the model:
    ```
    fun startAudioClassification() {
      if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        return
      }
      recorder.startRecording()
    }
    ```
1.  Set how frequently the model generates an inference from the audio
    clips by setting a fixed rate `interval` in the
    `ScheduledThreadPoolExecutor` object:
    ```
    executor = ScheduledThreadPoolExecutor(1)
    executor.scheduleAtFixedRate(
      classifyRunnable,
      0,
      interval,
      TimeUnit.MILLISECONDS)
    ```
1.  The `classifyRunnable` object in the code above executes the
    `AudioClassificationHelper.classifyAudio()` method, which loads the latest
    available audio data from the recorder and performs a prediction:
    ```
    private fun classifyAudio() {
      tensorAudio.load(recorder)
      val output = classifier.classify(tensorAudio)
      ...
    }
    ```

Caution: Do not run the ML model predictions on the main execution thread of
your application. Doing so can cause your app user interface to become slow or
unresponsive.

### Stop prediction processing

Make sure your app code stops doing audio classification when your app's audio
processing Fragment or Activity loses focus. Running a machine learning model
continuously has a significant impact on the battery life of an Android device.
Use the `onPause()` method of the Android activity or fragment associated with
the audio classification to stop audio recording and prediction processing.

To stop audio recording and classification:
-   Use the `AudioClassificationHelper.stopAudioClassification()` method to
    stop recording and the model execution, as shown below in the
    `AudioFragment` class:
    ```
    override fun onPause() {
      super.onPause()
      if (::audioHelper.isInitialized ) {
        audioHelper.stopAudioClassification()
      }
    }
    ```

## Handle model output

In your Android app, after you process an audio clip, the model produces a list
of predictions which your app code must handle by executing additional business
logic, displaying results to the user, or taking other actions.  The output of
any given TensorFlow Lite model varies in terms of the number of predictions it
produces (one or many), and the descriptive information for each prediction. In
the case of the models in the example app, the predictions are either a list of
recognized sounds or words. The AudioClassifier options object used in the code
example lets you set the maximum number of predictions with the
`setMaxResults()` method, as shown in [Initialize the ML
model](#Initialize_the_ML_model) section.

To get the prediction results from the model:

1.  Get the results of the
    [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier)
    object's `classify()` method and pass them to the listener object (code
    reference):
    ```
    private fun classifyAudio() {
      ...
      val output = classifier.classify(tensorAudio)
      listener.onResult(output[0].categories, inferenceTime)
    }
    ```
1.  Use the listener's onResult() function handle the output by executing
    business logic or displaying results to the user:
    ```
    private val audioClassificationListener = object : AudioClassificationListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        requireActivity().runOnUiThread {
          adapter.categoryList = results
          adapter.notifyDataSetChanged()
          fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)
        }
      }
    ```

The model used in this example generates a list of predictions with a label
for the classified sound or word, and a prediction score between 0 and 1 as a
Float representing the confidence of the prediction, with 1 being the highest
confidence rating. In general, predictions with a score below 50% (0.5) are
considered inconclusive. However, how you handle low-value prediction results is
up to you and the needs of your application.

Once the model has returned a set of prediction results, your application can
act on those predictions by presenting the result to your user or executing
additional logic. In the case of the example code, the application lists the
identified sounds or words in the app user interface.

## Next steps

You can find additional TensorFlow Lite models for audio processing on
[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding,audio-pitch-extraction,audio-event-classification,audio-stt)
and through the [Pre-trained models
guide](https://www.tensorflow.org/lite/models/trained) page. For more
information about implementing machine learning in your mobile application with
TensorFlow Lite, see the [TensorFlow Lite Developer
Guide](https://www.tensorflow.org/lite/guide).