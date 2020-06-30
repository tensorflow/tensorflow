# Integrate TensorFlow Lite models with metadata

[TensorFlow Lite metadata](../convert/metadata.md) contains a rich description
of what the model does and how to use the model. It can empower code generators,
such as the
[TensorFlow Lite Android code generator](#generate-code-with-tensorflow-lite-android-code-generator)
and the
[Android Studio ML Binding feature](#generate-code-with-android-studio-ml-model-binding),
to automatically generates the inference code for you. It can also be used to
configure your custom inference pipeline.

Browse
[TensorFlow Lite hosted models](https://www.tensorflow.org/lite/guide/hosted_models)
and [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) to download
pretrained models with metadata. All image models have been supported.

## Generate code with TensorFlow Lite Android code generator

Note: TensorFlow Lite wrapper code generator currently only supports Android.

For TensorFlow Lite model enhanced with [metadata](../convert/metadata.md),
developers can use the TensorFlow Lite Android wrapper code generator to create
platform specific wrapper code. The wrapper code removes the need to interact
directly with `ByteBuffer`. Instead, developers can interact with the TensorFlow
Lite model with typed objects such as `Bitmap` and `Rect`.

The usefulness of the code generator depend on the completeness of the
TensorFlow Lite model's metadata entry. Refer to the `<Codegen usage>` section
under relevant fields in
[metadata_schema.fbs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs),
to see how the codegen tool parses each field.

### Generate Wrapper Code

You will need to install the following tooling in your terminal:

```sh
pip install tflite-support
```

Once completed, the code generator can be used using the following syntax:

```sh
tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
```

The resulting code will be located in the destination directory. If you are
using [Google Colab](https://colab.research.google.com/) or other remote
environment, it maybe easier to zip up the result in a zip archive and download
it to your Android Studio project:

```python
## Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

## Kick off the download
from google.colab import files
files.download('classify_wrapper.zip')
```

### Using the generated code

#### Step 1: Import the generated code

Unzip the generated code if necessary into a directory structure. The root of
the generated code is assumed to be `SRC_ROOT`.

Open the Android Studio project where you would like to use the TensorFlow lite
model and import the generated module by: And File -> New -> Import Module ->
select `SRC_ROOT`

Using the above example, the directory and the module imported would be called
`classify_wrapper`.

#### Step 2: Update the app's `build.gradle` file

In the app module that will be consuming the generated library module:

Under the android section, add the following:

```build
aaptOptions {
   noCompress "tflite"
}
```

Under the dependencies section, add the following:

```build
implementation project(":classify_wrapper")
```

#### Step 3: Using the model

```java
// 1. Initialize the model
MyClassifierModel myImageClassifier = null;

try {
    myImageClassifier = new MyClassifierModel(this);
} catch (IOException io){
    // Error reading the model
}

if(null != myImageClassifier) {

    // 2. Set the input with a Bitmap called inputBitmap
    MyClassifierModel.Inputs inputs = myImageClassifier.createInputs();
    inputs.loadImage(inputBitmap));

    // 3. Run the model
    MyClassifierModel.Outputs outputs = myImageClassifier.run(inputs);

    // 4. Retrieve the result
    Map<String, Float> labeledProbability = outputs.getProbability();
}
```

### Accelerating model inference

The generated code provides a way for developers to accelerate their code
through the use of [delegates](../performance/delegates.md) and the number of
threads. These can be set when initiatizing the model object as it takes three
parameters:

*   **`Context`**: Context from the Android Activity or Service
*   (Optional) **`Device`**: TFLite acceleration delegate for example
    GPUDelegate or NNAPIDelegate
*   (Optional) **`numThreads`**: Number of threads used to run the model -
    default is one.

For example, to use a NNAPI delegate and up to three threads, you can initialize
the model like this:

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

### Troubleshooting

#### Getting 'java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed'

Under the app module that will uses the library module, insert the following
lines under the android section:

```build
aaptOptions {
   noCompress "tflite"
}
```

## Generate code with Android Studio ML Model Binding

[Android Studio ML Model Binding](https://developer.android.com/studio/preview/features#tensor-flow-lite-models)
allows you to directly import TensorFlow Lite models and use them in your
Android Studio projects. It generates easy-to-use classes so you can run your
model with less code and better type safety. See the
[introduction](https://developer.android.com/studio/preview/features#tensor-flow-lite-models)
for more details.

Note: Code generated by the TensorFlow Lite Android code generator may include
some latest API or experimental features, which can be a super set of the one
generated by the Android Studio ML Model Binding.

## Read the metadata from models

The Metadata Extractor library is a convinient tool to read the metadata and
associated files from a models across different platforms (see the
[Java version](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/support/metadata)
and the C++ version is coming soon). Users can also build their own metadata
extractor tool in other languages using the Flatbuffers library.

### Read the metadata in Java

Note: the Java Metadata Extractor library is available as an Android library
dependency: `org.tensorflow:tensorflow-lite-metadata`.

You can initialize a `MetadataExtractor` with a `ByteBuffer` that points to the
model:

```java
public MetadataExtractor(ByteBuffer buffer);
```

The `ByteBuffer` must remain unchanged for the whole lifetime of the
`MetadataExtractor`. The initialization may fail if the Flatbuffers file
identifier of the model metadata does not match the one of the metadata parser.
See [metadata versioning](../convert/metadata.md#metadata-versioning) for more
information.

As long as the file identifer is satisfied, the metadata extractor will not fail
when reading metadata generated from an old or a future scheme due to the
Flatbuffers forward and backwards compatibility mechanism. But fields from
future shcemas cannot be extracted by older metadata extractors. The
[minimum necessary parser version](../convert/metadata.md#the-minimum-necessary-metadata-parser-version)
of the metadata indicates the minimum version of metadata parser that can read
the metadata Flatbuffers in full. You can use the following method to verify if
the minimum necessary parser version is satisfied:

```java
public final boolean isMinimumParserVersionSatisfied();
```

It is allowed to pass in a model without metadata. However, invoking methods
that read from the metadata will cause runtime errors. You can check if a model
has metadata by invoking the method:

```java
public boolean hasMetadata();
```

`MetadataExtractor` provides convenient functions for you to get the
input/output tensors' metadata. For example,

```java
public int getInputTensorCount();
public TensorMetadata getInputTensorMetadata(int inputIndex);
public QuantizationParams getInputTensorQuantizationParams(int inputIndex);
public int[] getInputTensorShape(int inputIndex);
public int getoutputTensorCount();
public TensorMetadata getoutputTensorMetadata(int inputIndex);
public QuantizationParams getoutputTensorQuantizationParams(int inputIndex);
public int[] getoutputTensorShape(int inputIndex);
```

You can also read associated files through their names with the method:

```java
public InputStream getAssociatedFile(String fileName);
```

Though the
[TensorFlow Lite model schema](https://github.com/tensorflow/tensorflow/blob/aa7ff6aa28977826e7acae379e82da22482b2bf2/tensorflow/lite/schema/schema.fbs#L1075)
supports multiple subgraphs, the TFLite Interpreter only supports single
subgraph so far. Therefore, `MetadataExtractor` omits subgraph index as an input
in its methods.
