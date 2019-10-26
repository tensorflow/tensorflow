# TensorFlow Lite Android Support Library
Mobile application developers typically interact with typed objects such as bitmaps or primitives such as integers. However, the TensorFlow Lite (TFLite) Interpreter that runs the on-device machine learning model uses tensors in the form of ByteBuffer, which can be difficult to debug and manipulate. The TFLite Android Support Library is designed to help with conversion of input and output of TFLite models, as well as with making the TFLite interpreter easier to use. We welcome feedback from the community as we develop this support library, especially around:
* Use-cases we should support including data types and operations
* Ease of use - does the APIs make sense to the community

## Table of Contents
* [Getting Started](#getting-started)
    * [Import Gradle dependency and other settings](#Import-Gradle-dependency-and-other-settings)
    * [Basic image manipulation and conversion](#Basic-image-manipulation-and-conversion)
    * [Create output objects and run the model](#Create-output-objects-and-run-the-model)
    * [Accessing the result](#Accessing-the-result)
    * [Optional: Mapping results to labels](#Optional:-Mapping-results-to-labels)
* [Current use-case coverage](#Current-use-case-coverage)
* [ImageProcessor Architecture](#ImageProcessor-Architecture)
* [Quantization](#Quantization)


## Getting Started
### Import Gradle dependency and other settings
Copy the tflite model file to the assets directory for the Android module where the model will be run. Specify that the tflite file should not be compressed and add the dependency in the module’s build.gradle file:

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
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'
}
```

### Basic image manipulation and conversion
The TFLite support library has a suite of basic image manipulation methods such as crop and resize. To use it, create an `ImagePreprocessor` and add the required operations. In addition, to convert the image to tensors needed for TFLite interpreter, create a `TensorImage` to  be used as input:

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

// Create a TensorImage object, this creates the tensor the TFLite
// interpreter needs
TensorImage tImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tImage.load(bitmap);
tImage = imageProcessor.process(tImage);
```
### Create output objects and run the model
Before running the model, we need to create the container objects that will store the result:

```java
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create containers for the result and specified that this is a quantized model,
// hence, the DataType is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);

// Put the array into a map. If there are multiple objects being returned
// the outputs map can contain multiple objects
Map<Integer, Object> outputs = new HashMap<>();
outputs.put(0, probabilityBuffer.getBuffer());

Loading the model and running inference:
import org.tensorflow.lite.support.model.Model;

// Initialise the model
Model model = null;
try{

    model = new Model.Builder(this, "mobilenet_v1_1.0_224_quant.tflite")
        .build();

} catch (IOException e){
    Log.e("tfliteSupport", "Error reading model", e);
}

// Running inference
if(null != model)
    model.run(new Object[]{tImage.getBuffer()}, outputs);
```

### Accessing the result
Developers can access the output directly through `probabilityBuffer.getFloatArray()`. If the model is quantized, remember to convert the result. For MobileNet quantized model, developer needs to divide each element by 255 to obtain the probability ranging from 0 (least likely) to 1 (most likely) for each category.

### Optional: Mapping results to labels
Developers can also optionally map results to labels. First, copy the text file containing labels into the module’s assets directory. Next, load the label file using:

```java
import org.tensorflow.lite.support.common.FileUtil;

final String ASSOCIATED_AXIS_LABELS = "labels.txt";
List<String> associatedAxisLabels = null;

try{
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e){
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

Associating the probabilities to category labels:
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
The current experimental version covers:
 * common data types (float, uint8, images and array of these objects) as inputs and outputs of tflite models.
 * basic image operations (crop image, resize and rotate)
 * quantized and float models

This should cover a wide range of numerical and image based models. In addition to improving on the current offering, we aim to improve support for string related operations in the next step.

## ImageProcessor Architecture
The design of the `ImageProcessor` allowed the image manipulation operations to be defined up front and optimised during the build process. The `ImageProcessor` currently supports three basic preprocessing operations:

```java
int width = bitmap.getWidth();
int height = bitmap.getHeight();

int size = height > width ? width : height;

ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        // Center crop the image to the largest square possible
        .add(new ResizeWithCropOrPadOp(size , size))
        // Resize using Bilinear or Nearest neighbour
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR));
        // Rotation counter-clockwise in 90 degree increments
        .add(new Rot90Op(rotateDegrees/90))
        .build();
```

The eventual goal of the support library is to support all [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) transformations. This means the transformation will be the same as TensorFlow and the implementation will be independent of the operating system.

Developers are also welcome to create custom processors. It is important in these cases to be aligned with the training process - i.e. the same preprocessing should apply to both training and inference to increase reproducibility.

## Quantization
When initiating input or output objects such as `TensorImage` or `TensorBuffer`, developer will need to specify whether they are to be quantized objects by specifying their type to be `DataType.UINT8` or `DataType.FLOAT32`.

`TensorProcessor` can be used to quantize input tensors or dequantize output tensors. For example, when processing a quantized output `TensorBuffer`, developer can call the following to dequantized the result to float probability between 0 and 1:

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

// Post-processor which dequantize the result
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer)
```
