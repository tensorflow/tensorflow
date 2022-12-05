# Integrate image classifiers

Image classification is a common use of machine learning to identify what an
image represents. For example, we might want to know what type of animal appears
in a given picture. The task of predicting what an image represents is called
*image classification*. An image classifier is trained to recognize various
classes of images. For example, a model might be trained to recognize photos
representing three different types of animals: rabbits, hamsters, and dogs. See
the
[image classification overview](../../examples/image_classification/overview)
for more information about image classifiers.

Use the Task Library `ImageClassifier` API to deploy your custom image
classifiers or pretrained ones into your mobile apps.

## Key features of the ImageClassifier API

*   Input image processing, including rotation, resizing, and color space
    conversion.

*   Region of interest of the input image.

*   Label map locale.

*   Score threshold to filter results.

*   Top-k classification results.

*   Label allowlist and denylist.

## Supported image classifier models

The following models are guaranteed to be compatible with the `ImageClassifier`
API.

*   Models created by
    [TensorFlow Lite Model Maker for Image Classification](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

*   The
    [pretrained image classification models on TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1).

*   Models created by
    [AutoML Vision Edge Image Classification](https://cloud.google.com/vision/automl/docs/edge-quickstart).

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

See the
[Image Classification reference app](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)
for an example of how to use `ImageClassifier` in an Android app.

### Step 1: Import Gradle dependency and other settings

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

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Step 2: Using the model

```java
// Initialization
ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ImageClassifier imageClassifier =
    ImageClassifier.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

See the
[source code and javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java)
for more options to configure `ImageClassifier`.

## Run inference in iOS

### Step 1: Install the dependencies

The Task Library supports installation using CocoaPods. Make sure that CocoaPods
is installed on your system. Please see the
[CocoaPods installation guide](https://guides.cocoapods.org/using/getting-started.html#getting-started)
for instructions.

Please see the
[CocoaPods guide](https://guides.cocoapods.org/using/using-cocoapods.html) for
details on adding pods to an Xcode project.

Add the `TensorFlowLiteTaskVision` pod in the Podfile.

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskVision'
end
```

Make sure that the `.tflite` model you will be using for inference is present in
your app bundle.

### Step 2: Using the model

#### Swift

```swift
// Imports
import TensorFlowLiteTaskVision

// Initialization
guard let modelPath = Bundle.main.path(forResource: "birds_V1",
                                            ofType: "tflite") else { return }

let options = ImageClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try ImageClassifier.classifier(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "sparrow.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let classificationResults = try classifier.classify(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"birds_V1" ofType:@"tflite"];

TFLImageClassifierOptions *options =
    [[TFLImageClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLImageClassifier *classifier = [TFLImageClassifier imageClassifierWithOptions:options
                                                                          error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"sparrow.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLClassificationResult *classificationResult =
    [classifier classifyWithGMLImage:gmlImage error:nil];
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifier.h)
for more options to configure `TFLImageClassifier`.

## Run inference in Python

### Step 1: Install the pip package

```
pip install tflite-support
```

### Step 2: Using the model

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_classifier.py)
for more options to configure `ImageClassifier`.

## Run inference in C++

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h

std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_classifier.h)
for more options to configure `ImageClassifier`.

## Example results

Here is an example of the classification results of a
[bird classifier](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3).

<img src="images/sparrow.jpg" alt="sparrow" width="50%">

```
Results:
  Rank #0:
   index       : 671
   score       : 0.91406
   class name  : /m/01bwb9
   display name: Passer domesticus
  Rank #1:
   index       : 670
   score       : 0.00391
   class name  : /m/01bwbt
   display name: Passer montanus
  Rank #2:
   index       : 495
   score       : 0.00391
   class name  : /m/0bwm6m
   display name: Passer italiae
```

Try out the simple
[CLI demo tool for ImageClassifier](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-classifier)
with your own model and test data.

## Model compatibility requirements

The `ImageClassifier` API expects a TFLite model with mandatory
[TFLite Model Metadata](../../models/convert/metadata). See examples of creating
metadata for image classifiers using the
[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#image_classifiers).

The compatible image classifier models should meet the following requirements:

*   Input image tensor (kTfLiteUInt8/kTfLiteFloat32)

    -   image input of size `[batch x height x width x channels]`.
    -   batch inference is not supported (`batch` is required to be 1).
    -   only RGB inputs are supported (`channels` is required to be 3).
    -   if type is kTfLiteFloat32, NormalizationOptions are required to be
        attached to the metadata for input normalization.

*   Output score tensor (kTfLiteUInt8/kTfLiteFloat32)

    -   with `N` classes and either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1
        x 1 x N]`
    -   optional (but recommended) label map(s) as AssociatedFile-s with type
        TENSOR_AXIS_LABELS, containing one label per line. See the
        [example label file](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt).
        The first such AssociatedFile (if any) is used to fill the `label` field
        (named as `class_name` in C++) of the results. The `display_name` field
        is filled from the AssociatedFile (if any) whose locale matches the
        `display_names_locale` field of the `ImageClassifierOptions` used at
        creation time ("en" by default, i.e. English). If none of these are
        available, only the `index` field of the results will be filled.
