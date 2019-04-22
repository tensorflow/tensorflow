# iOS quickstart

To get started with TensorFlow Lite on iOS, we recommend exploring the following
example.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS
image classification example</a>

For an explanation of the source code, you should also read
[TensorFlow Lite iOS image classification](https://www.tensorflow.org/lite/models/image_classification/ios).

This example app uses
[image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
to continuously classify whatever it sees from the device's rear-facing camera.
The application must be run on an iOS device.

Inference is performed using the TensorFlow Lite C++ API. The demo app
classifies frames in real-time, displaying the top most probable
classifications. It allows the user to choose between a floating point or
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
model, select the thread count, and decide whether to run on CPU, GPU, or via
[NNAPI](https://developer.android.com/ndk/guides/neuralnetworks).

Note: Additional iOS applications demonstrating TensorFlow Lite in a variety of
use cases are available in [Examples](https://www.tensorflow.org/lite/examples).

## Build in Xcode

To build the example in Xcode, follow the instructions in
[README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/README.md).

## Create your own iOS app

To get started quickly writing your own iOS code, we recommend using our
[iOS image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)
as a starting point.

The following sections contain some useful information for working with
TensorFlow Lite on iOS.

### Use TensorFlow Lite from Objective-C and Swift

The example app provides an Objective-C wrapper on top of the C++ Tensorflow
Lite library. This wrapper is required because currently there is no
interoperability between Swift and C++. The wrapper is exposed to Swift via
bridging so that the Tensorflow Lite methods can be called from Swift.

The wrapper is located in
[TensorflowLiteWrapper](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/TensorflowLiteWrapper).
It is not tightly coupled with the example code, so you can use it in your own
iOS apps. It exposes the following interface:

```objectivec
@interface TfliteWrapper : NSObject

/**
 This method initializes the TfliteWrapper with the specified model file.
 */
- (instancetype)initWithModelFileName:(NSString *)fileName;

/**
 This method initializes the interpreter of TensorflowLite library with the specified model file
 that performs the inference.
 */
- (BOOL)setUpModelAndInterpreter;

/**
 This method gets a reference to the input tensor at an index.
 */
- (uint8_t *)inputTensorAtIndex:(int)index;

/**
 This method performs the inference by invoking the interpreter.
 */
- (BOOL)invokeInterpreter;

/**
 This method gets the output tensor at a specified index.
 */
- (uint8_t *)outputTensorAtIndex:(int)index;

/**
 This method sets the number of threads used by the interpreter to perform inference.
 */
- (void)setNumberOfThreads:(int)threadCount;

@end
```

To use these files in your own iOS app, copy them into your Xcode project.

Note: When you add an Objective-C file to an existing Swift app (or vice versa),
Xcode will prompt you to create a *bridging header* file to expose the files to
Swift. In the example project, this file is named
[`ImageClassification-Bridging-Header.h`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/TensorflowLiteWrapper/ImageClassification-Bridging-Header.h).
For more information, see Apple's
[Importing Objective-C into Swift](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_objective-c_into_swift){: .external}
documentation.
