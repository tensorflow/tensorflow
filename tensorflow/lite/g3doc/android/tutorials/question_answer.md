# Answering questions with Android

![Question answering example app in Android](../../examples/bert_qa/images/screenshot.gif){: .attempt-right width="250px"}

This tutorial shows you how to build an Android application using TensorFlow
Lite to provide answers to questions structured in natural language text. The
[example application](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android)
uses the *BERT question answerer*
([`BertQuestionAnswerer`](../../inference_with_metadata/task_library/bert_question_answerer))
API within the
[Task library for natural language (NL)](../../inference_with_metadata/task_library/overview#supported_tasks)
to enable Question answering machine learning models. The application is
designed for a physical Android device but can also run on a device emulator.

If you are updating an existing project, you can use the example application as
a reference or template. For instructions on how to add question answering to an
existing application, refer to
[Updating and modifying your application](#modify_applications).

## Question answering overview

*Question answering* is the machine learning task of answering questions posed
in natural language. A trained question answering model receives a text passage
and question as input, and attempts to answer the question based on its
interpretation of the information within the passage.

A Question answering model is trained on a question answering dataset, which
consists of a reading comprehension dataset along with question-answer pairs
based on different segments of text.

For more information on how the models in this tutorial are generated, refer to
the
[BERT Question Answer with TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)
tutorial.

## Models and dataset

The example app uses the Mobile BERT Q&A
([`mobilebert`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1))
model, which is a lighter and faster version of
[BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations
from Transformers). For more information on `mobilebert`, see the
[MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)
research paper.

The `mobilebert` model was trained using the Stanford Question Answering Dataset
([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) dataset, a reading
comprehension dataset consisting of articles from Wikipedia and a set of
question-answer pairs for each article.

## Setup and run the example app

To setup the question answering application, download the example app from
[GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android)
and run it using [Android Studio](https://developer.android.com/studio/).

### System requirements

*   **[Android Studio](https://developer.android.com/studio/index.html)**
    version 2021.1.1 (Bumblebee) or higher.
*   Android SDK version 31 or higher
*   Android device with a minimum OS version of SDK 21 (Android 7.0 - Nougat)
    with
    [developer mode](https://developer.android.com/studio/debug/dev-options)
    enabled, or an Android Emulator.

### Get the example code

Create a local copy of the example code. You will use this code to create a
project in Android Studio and run the example application.

To clone and setup the example code:

1.  Clone the git repository
    <pre class="devsite-click-to-copy">
    git clone https://github.com/tensorflow/examples.git
    </pre>
2.  Optionally, configure your git instance to use sparse checkout, so you have only
    the files for the question answering example app:
    <pre class="devsite-click-to-copy">
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/bert_qa/android
    </pre>

### Import and run the project

Create a project from the downloaded example code, build the project, and then
run it.

To import and build the example code project:

1.  Start [Android Studio](https://developer.android.com/studio).
1.  From the Android Studio, select **File > New > Import Project**.
1.  Navigate to the example code directory containing the build.gradle file
    (`.../examples/lite/examples/bert_qa/android/build.gradle`) and select that
    directory.
1.  If Android Studio requests a Gradle Sync, choose OK.
1.  Ensure that your Android device is connected to your computer and developer
    mode is enabled. Click the green `Run` arrow.

If you select the correct directory, Android Studio creates a new project and
builds it. This process can take a few minutes, depending on the speed of your
computer and if you have used Android Studio for other projects. When the build
completes, the Android Studio displays a `BUILD SUCCESSFUL` message in the
**Build Output** status panel.

To run the project:

1.  From Android Studio, run the project by selecting **Run > Run…**.
1.  Select an attached Android device (or emulator) to test the app.

### Using the application

After running the project in Android Studio, the application automatically opens
on the connected device or device emulator.

To use the Question answerer example app:

1.  Choose a topic from the list of subjects.
1.  Choose a suggested question or enter your own in the text box.
1.  Toggle the orange arrow to run the model.

The application attempts to identify the answer to the question from the passage
text. If the model detects an answer within the passage, the application
highlights the relevant span of text for the user.

You now have a functioning question answering application. Use the following
sections to better understand how the example application works, and how to
implement question answering features in your production applications:

*   [How the application works](#how_it_works) - A walkthrough of the structure
    and key files of the example application.

*   [Modify your application](#modify_applications) - Instructions on adding
    question answering to an existing application.

## How the example app works {:#how_it_works}

The application uses the `BertQuestionAnswerer` API within the
[Task library for natural language (NL)](../../inference_with_metadata/task_library/overview#supported_tasks)
package. The MobileBERT model was trained using the TensorFlow Lite
[Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).
The application runs on CPU by default, with the option of hardware acceleration
using the GPU or NNAPI delegate.

The following files and directories contain the crucial code for this
application:

*   [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt) -
    Initializes the question answerer and handles the model and delegate
    selection.
*   [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt) -
    Handles and formats the results.
*   [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/MainActivity.kt) -
    Provides the organizing logic of the app.

## Modify your application {:#modify_applications}

The following sections explain the key steps to modify your own Android app to
run the model shown in the example app. These instructions use the example app
as a reference point. The specific changes needed for your own app may vary from
the example app.

### Open or create an Android project

You need an Android development project in Android Studio to follow along with
the rest of these instructions. Follow the instructions below to open an
existing project or create a new one.

To open an existing Android development project:

*   In Android Studio, select *File > Open* and select an existing project.

To create a basic Android development project:

*   Follow the instructions in Android Studio to
    [Create a basic project](https://developer.android.com/studio/projects/create-project).

For more information on using Android Studio, refer to the
[Android Studio documentation](https://developer.android.com/studio/intro).

### Add project dependencies

In your own application, add specific project dependencies to run TensorFlow
Lite machine learning models and access utility functions. These functions
convert data such as strings into a tensor data format that can be processed by
the model. The following instructions explain how to add the required project
and module dependencies to your own Android app project.

To add module dependencies:

1.  In the module that uses TensorFlow Lite, update the module's `build.gradle`
    file to include the following dependencies.

    In the example application, the dependencies are located in
    [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/build.gradle):

    ```
    dependencies {
      ...
      // Import tensorflow library
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'

      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    The project must include the Text task library
    (`tensorflow-lite-task-text`).

    If you want to modify this app to run on a graphics processing unit (GPU),
    the GPU library (`tensorflow-lite-gpu-delegate-plugin`) provides the
    infrastructure to run the app on GPU, and Delegate (`tensorflow-lite-gpu`)
    provides the compatibility list.

1.  In Android Studio, sync the project dependencies by selecting: **File > Sync
    Project with Gradle Files**.

### Initialize the ML models {:#initialize_models}

In your Android app, you must initialize the TensorFlow Lite machine learning
model with parameters before running predictions with the model.

A TensorFlow Lite model is stored as a `*.tflite` file. The model file contains
the prediction logic and typically includes
[metadata](../../models/convert/metadata) about how to interpret prediction
results. Typically, model files are stored in the `src/main/assets` directory of
your development project, as in the code example:

-   `<project>/src/main/assets/mobilebert_qa.tflite`

Note: The example app uses a
[`download_model.gradle`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/download_models.gradle)
file to download the
[mobilebert_qa](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer)
model and
[passage text](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/contents_from_squad.json)
at build time. This approach is not required for a production app.

For convenience and code readability, the example declares a companion object
that defines the settings for the model.

To initialize the model in your app:

1.  Create a companion object to define the settings for the model. In the
    example application, this object is located in
    [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L100-L106):

    ```
    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
    ```

1.  Create the settings for the model by building a `BertQaHelper` object, and
    construct a TensorFlow Lite object with `bertQuestionAnswerer`.

    In the example application, this is located in the
    `setupBertQuestionAnswerer()` function within
    [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L41-L76):

    ```
    class BertQaHelper(
        ...
    ) {
        ...
        init {
            setupBertQuestionAnswerer()
        }

        fun clearBertQuestionAnswerer() {
            bertQuestionAnswerer = null
        }

        private fun setupBertQuestionAnswerer() {
            val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
            ...
            val options = BertQuestionAnswererOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .build()

            try {
                bertQuestionAnswerer =
                    BertQuestionAnswerer.createFromFileAndOptions(context, BERT_QA_MODEL, options)
            } catch (e: IllegalStateException) {
                answererListener
                    ?.onError("Bert Question Answerer failed to initialize. See error logs for details")
                Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            }
        }
        ...
        }
    ```

### Enable hardware acceleration (optional) {:#hardware_acceleration}

When initializing a TensorFlow Lite model in your app, you should consider using
hardware acceleration features to speed up the prediction calculations of the
model. TensorFlow Lite
[delegates](https://www.tensorflow.org/lite/performance/delegates) are software
modules that accelerate the execution of machine learning models using
specialized processing hardware on a mobile device, such as graphics processing
unit (GPUs) or tensor processing units (TPUs).

To enable hardware acceleration in your app:

1.  Create a variable to define the delegate that the application will use. In
    the example application, this variable is located early in
    [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L31):

    ```
    var currentDelegate: Int = 0
    ```

1.  Create a delegate selector. In the example application, the delegate
    selector is located in the `setupBertQuestionAnswerer` function within
    [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L48-L62):

    ```
    when (currentDelegate) {
        DELEGATE_CPU -> {
            // Default
        }
        DELEGATE_GPU -> {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                baseOptionsBuilder.useGpu()
            } else {
                answererListener?.onError("GPU is not supported on this device")
            }
        }
        DELEGATE_NNAPI -> {
            baseOptionsBuilder.useNnapi()
        }
    }
    ```

Using delegates for running TensorFlow Lite models is recommended, but not
required. For more information about using delegates with TensorFlow Lite, see
[TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates).

### Prepare data for the model

In your Android app, your code provides data to the model for interpretation by
transforming existing data such as raw text into a
[Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor)
data format that can be processed by your model. The Tensor you pass to a model
must have specific dimensions, or shape, that matches the format of data used to
train the model. This question answering app accepts
[strings](https://developer.android.com/reference/java/lang/String.html) as
inputs for both the text passage and question. The model does not recognize
special characters and non-English words.

To provide passage text data to the model:

1.  Use the `LoadDataSetClient` object to load the passage text data to the app.
    In the example application, this is located in
    [LoadDataSetClient.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/dataset/LoadDataSetClient.kt#L25-L45)

    ```
    fun loadJson(): DataSet? {
        var dataSet: DataSet? = null
        try {
            val inputStream: InputStream = context.assets.open(JSON_DIR)
            val bufferReader = inputStream.bufferedReader()
            val stringJson: String = bufferReader.use { it.readText() }
            val datasetType = object : TypeToken<DataSet>() {}.type
            dataSet = Gson().fromJson(stringJson, datasetType)
        } catch (e: IOException) {
            Log.e(TAG, e.message.toString())
        }
        return dataSet
    }
    ```

1.  Use the `DatasetFragment` object to list the titles for each passage of text
    and start the **TFL Question and Answer** screen. In the example
    application, this is located in
    [DatasetFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetFragment.kt):

    ```
    class DatasetFragment : Fragment() {
        ...
        override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
            super.onViewCreated(view, savedInstanceState)
            val client = LoadDataSetClient(requireActivity())
            client.loadJson()?.let {
                titles = it.getTitles()
            }
            ...
        }
       ...
    }
    ```

1.  Use the `onCreateViewHolder` function within the `DatasetAdapter` object to
    present the titles for each passage of text. In the example application,
    this is located in
    [DatasetAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetAdapter.kt):

    ```
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }
    ```

To provide user questions to the model:

1.  Use the `QaAdapter` object to provide the question to the model. In the
    example application, this is located in
    [QaAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaAdapter.kt):

    ```
    class QaAdapter(private val question: List<String>, private val select: (Int) -> Unit) :
      RecyclerView.Adapter<QaAdapter.ViewHolder>() {

      inner class ViewHolder(private val binding: ItemQuestionBinding) :
          RecyclerView.ViewHolder(binding.root) {
          init {
              binding.tvQuestionSuggestion.setOnClickListener {
                  select.invoke(adapterPosition)
              }
          }

          fun bind(question: String) {
              binding.tvQuestionSuggestion.text = question
          }
      }
      ...
    }
    ```

### Run predictions

In your Android app, once you have initialized a
[BertQuestionAnswerer](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer)
object, you can begin inputting questions in the form of natural language text
to the model. The model attempts to identify the answer within the text passage.

To run predictions:

1.  Create an `answer` function, which runs the model and measures the time
    taken to identify the answer (`inferenceTime`). In the example application,
    the `answer` function is located in
    [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L78-L98):

    ```
    fun answer(contextOfQuestion: String, question: String) {
        if (bertQuestionAnswerer == null) {
            setupBertQuestionAnswerer()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val answers = bertQuestionAnswerer?.answer(contextOfQuestion, question)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        answererListener?.onResults(answers, inferenceTime)
    }
    ```

1.  Pass the results from `answer` to the listener object.

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

### Handle model output

After you input a question, the model provides a maximum of five possible
answers within the passage.

To get the results from the model:

1.  Create an `onResult` function for the listener object to handle the output.
    In the example application, the listener object is located in
    [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L92-L98)

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

1.  Highlight sections of the passage based on the results. In the example
    application, this is located in
    [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt#L199-L208):

    ```
    override fun onResults(results: List<QaAnswer>?, inferenceTime: Long) {
        results?.first()?.let {
            highlightAnswer(it.text)
        }

        fragmentQaBinding.tvInferenceTime.text = String.format(
            requireActivity().getString(R.string.bottom_view_inference_time),
            inferenceTime
        )
    }
    ```

Once the model has returned a set of results, your application can act on those
predictions by presenting the result to your user or executing additional logic.

## Next steps

*   Train and implement the models from scratch with the
    [Question Answer with TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)
    tutorial.
*   Explore more
    [text processing tools for TensorFlow](https://www.tensorflow.org/text).
*   Download other BERT models on
    [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1).
*   Explore various uses of TensorFlow Lite in the [examples](../../examples).
