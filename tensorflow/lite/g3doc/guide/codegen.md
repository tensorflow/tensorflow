# Generate code from TensorFlow Lite metadata

Note: TensorFlow Lite wrapper code generator is in experimental (beta) phase and
it currently only supports Android.

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

## Generate Wrapper Code

You will need to install the following tooling in your terminal:

```
pip install tflite-support
```

Once completed, the code generator can be used using the following syntax:

```
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

## Using the generated code

### Step 1: Import the generated code

Unzip the generated code if necessary into a directory structure. The root of
the generated code is assumed to be `SRC_ROOT`.

Open the Android Studio project where you would like to use the TensorFlow lite
model and import the generated module by: And File -> New -> Import Module ->
select `SRC_ROOT`

Using the above example, the directory and the module imported would be called
`classify_wrapper`.

### Step 2: Update the app's `build.gradle` file

In the app module that will be consuming the generated library module:

Under the android section, add the following:

```java
aaptOptions {
   noCompress "tflite"
}
```

Under the dependencies section, add the following:

```java
implementation project(":classify_wrapper")
```

### Step 3: Using the model

```java
// 1. Initialize the Model
MyClassifierModel myImageClassifier = null;

try {
    myImageClassifier = new MyClassifierModel(this);
} catch (IOException io){
    // Error reading the model
}

if(null != myImageClassifier) {

    // 2. Setting the input with a Bitmap called inputBitmap
    MyClassifierModel.Inputs inputs = myImageClassifier.createInputs();
    inputs.loadImage(inputBitmap));

    // 3. Running the model
    MyClassifierModel.Outputs outputs = myImageClassifier.run(inputs);

    // 4. Retrieving the result
    Map<String, Float> labeledProbability = outputs.getProbability();
}
```

## Accelerating model inference

The generated code provides a way for developers to accelerate their code
through the use of [delegates](../performance/delegates.md) and the number of
threads. These can be set when initiatizing the model object as it takes three
parameters:

*   **`Context`**: Context from the Android Activity or Service
*   (Optional) **`Device`**: TFLite acceleration delegate for example
    GPUDelegate or NNAPIDelegate
*   (Optional) **`numThreads`**: Number of threads used to run the model -
    default is one.

For example, to use a NNAPI delegate and up to three threads, you can initiate
the model like this:

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

## Troubleshooting

### Getting 'java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed'

Under the app module that will uses the library module, insert the following
lines under the android section:

```java
aaptOptions {
   noCompress "tflite"
}
```
