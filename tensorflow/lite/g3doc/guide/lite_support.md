# Process input and output data with the TensorFlow Lite Support Library

Note: TensorFlow Lite Support Library currently only supports Android.

Mobile application developers typically interact with typed objects such as
bitmaps or primitives such as integers. However, the TensorFlow Lite Interpreter
that runs the on-device machine learning model uses tensors in the form of
ByteBuffer, which can be difficult to debug and manipulate. The
[TensorFlow Lite Android Support Library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/support/java)
is designed to help process the input and output of TensorFlow Lite models, and
make the TensorFlow Lite interpreter easier to use.

## Getting Started

### Import Gradle dependency and other settings

Copy the `.tflite` model file to the assets directory of the Android module
where the model will be run. Specify that the file should not be compressed, and
add the TensorFlow Lite library to the module’s `build.gradle` file:

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import tflite dependencies
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    // The GPU delegate library is optional. Depend on it as needed.
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'
}
```

### Basic image manipulation and conversion

The TensorFlow Lite Support Library has a suite of basic image manipulation
methods such as crop and resize. To use it, create an `ImagePreprocessor` and
add the required operations. To convert the image into the tensor format
required by the TensorFlow Lite interpreter, create a `TensorImage` to be used
as input:

```java
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

// Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build();

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
TensorImage tImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tImage.load(bitmap);
tImage = imageProcessor.process(tImage);
```

`DataType` of a tensor can be read through the
[metadata exractor library](../guide/codegen.md#read-the-metadata-from-models)
as well as other model information.

### Create output objects and run the model

Before running the model, we need to create the container objects that will
store the result:

```java
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

Loading the model and running inference:

```java
import org.tensorflow.lite.support.model.Model;

// Initialise the model
try{
    MappedByteBuffer tfliteModel
        = FileUtil.loadMappedFile(activity,
            "mobilenet_v1_1.0_224_quant.tflite");
    Interpreter tflite = new Interpreter(tfliteModel)
} catch (IOException e){
    Log.e("tfliteSupport", "Error reading model", e);
}

// Running inference
if(null != tflite) {
    tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
}
```

### Accessing the result

Developers can access the output directly through
`probabilityBuffer.getFloatArray()`. If the model produces a quantized output,
remember to convert the result. For the MobileNet quantized model, the developer
needs to divide each output value by 255 to obtain the probability ranging from
0 (least likely) to 1 (most likely) for each category.

### Optional: Mapping results to labels

Developers can also optionally map the results to labels. First, copy the text
file containing labels into the module’s assets directory. Next, load the label
file using the following code:

```java
import org.tensorflow.lite.support.common.FileUtil;

final String ASSOCIATED_AXIS_LABELS = "labels.txt";
List<String> associatedAxisLabels = null;

try {
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e) {
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

The following snippet demonstrates how to associate the probabilities with
category labels:

```java
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.label.TensorLabel;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

if (null != associatedAxisLabels) {
    // Map of labels and their corresponding probability
    TensorLabel labels = new TensorLabel(associatedAxisLabels,
        probabilityProcessor.process(probabilityBuffer));

    // Create a map to access the result based on label
    Map<String, Float> floatMap = labels.getMapWithFloatValue();
}
```

## Current use-case coverage

The current version of the TensorFlow Lite Support Library covers:

*   common data types (float, uint8, images and array of these objects) as
    inputs and outputs of tflite models.
*   basic image operations (crop image, resize and rotate).
*   normalization and quantization
*   file utils

Future versions will improve support for text-related applications.

## ImageProcessor Architecture

The design of the `ImageProcessor` allowed the image manipulation operations to
be defined up front and optimised during the build process. The `ImageProcessor`
currently supports three basic preprocessing operations:

```java
int width = bitmap.getWidth();
int height = bitmap.getHeight();

int size = height > width ? width : height;

ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        // Center crop the image to the largest square possible
        .add(new ResizeWithCropOrPadOp(size, size))
        // Resize using Bilinear or Nearest neighbour
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR));
        // Rotation counter-clockwise in 90 degree increments
        .add(new Rot90Op(rotateDegrees / 90))
        .add(new NormalizeOp(127.5, 127.5))
        .add(new QuantizeOp(128.0, 1/128.0))
        .build();
```

See more details
[here](../convert/metadata.md#normalization-and-quantization-parameters) about
normalization and quantization.

The eventual goal of the support library is to support all
[`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image)
transformations. This means the transformation will be the same as TensorFlow
and the implementation will be independent of the operating system.

Developers are also welcome to create custom processors. It is important in
these cases to be aligned with the training process - i.e. the same
preprocessing should apply to both training and inference to increase
reproducibility.

## Quantization

When initiating input or output objects such as `TensorImage` or `TensorBuffer`
you need to specify their types to be `DataType.UINT8` or `DataType.FLOAT32`.

```java
TensorImage tImage = new TensorImage(DataType.UINT8);
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

The `TensorProcessor` can be used to quantize input tensors or dequantize output
tensors. For example, when processing a quantized output `TensorBuffer`, the
developer can use `DequantizeOp` to dequantize the result to a floating point
probability between 0 and 1:

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255.0)).build();
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);
```

The quantization parameters of a tensor can be read through the
[metadata exractor library](../guide/codegen.md#read-the-metadata-from-models).
