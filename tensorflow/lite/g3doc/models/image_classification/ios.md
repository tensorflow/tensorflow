# TensorFlow Lite iOS image classification example

This document walks through the code of a simple iOS mobile application that
demonstrates [image classification](overview.md) using the device camera.

The application code is located in the
[Tensorflow examples](https://github.com/tensorflow/examples) repository, along
with instructions for building and deploying the app.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">Example
application</a>

## Explore the code

We're now going to walk through the most important parts of the sample code.

This example is written in both Swift and Objective-C. All application
functionality, image processing, and results formatting is developed in Swift.
Objective-C is used via
[bridging](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_objective-c_into_swift)
to make the TensorFlow Lite C++ framework calls.

### Get camera input

The main logic of this app is in the Swift source file
[`ViewController.swift`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/ViewControllers/ViewController.swift).

The app's main view is represented by the `ViewController` class, which we
extend with functionality from `CameraFeedManagerDelegate`, a class created to
handle a camera feed. To run inference on the feed, we implement the `didOutput`
method, which is called whenever a frame is available from the camera.

Our implementation of `didOutput` includes a call to the `runModel` method of a
`ModelDataHandler` instance. As we will see below, this class gives us access to
the TensorFlow Lite interpreter and the image classification model we are using.

```swift
extension ViewController: CameraFeedManagerDelegate {

  func didOutput(pixelBuffer: CVPixelBuffer) {

    // Run the live camera pixelBuffer through TensorFlow to get the result
    let currentTimeMs = Date().timeIntervalSince1970 * 1000

    guard  (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else {
      return
    }

    previousInferenceTimeMs = currentTimeMs
    result = modelDataHandler?.runModel(onFrame: pixelBuffer)

    DispatchQueue.main.async {

      let resolution = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))

      // Display results by handing off to the InferenceViewController
      self.inferenceViewController?.inferenceResult = self.result
      self.inferenceViewController?.resolution = resolution
      self.inferenceViewController?.tableView.reloadData()

    }
  }
...
```

### TensorFlow Lite wrapper

The app uses TensorFlow Lite's C++ library via an Objective-C wrapper defined in
[`TfliteWrapper.h`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/TensorFlowLiteWrapper/TfliteWrapper.h)
and
[`TfliteWrapper.mm`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/TensorFlowLiteWrapper/TfliteWrapper.mm).

This wrapper is required because currently there is no interoperability between
Swift and C++. The wrapper is exposed to Swift via bridging so that the
Tensorflow Lite methods can be called from Swift.

### ModelDataHandler

The Swift class `ModelDataHandler`, defined by
[`ModelDataHandler.swift`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/ModelDataHandler/ModelDataHandler.swift),
handles all data preprocessing and makes calls to run inference on a given frame
through the TfliteWrapper. It then formats the inferences obtained and returns
the top N results for a successful inference.

The following sections show how this works.

#### Initialization

The method `init` instantiates a `TfliteWrapper` and loads the supplied model
and labels files from disk.

```swift
init?(modelFileName: String, labelsFileName: String, labelsFileExtension: String) {

  // Initializes TFliteWrapper and based on the setup result of interpreter, initializes the object of this class
  self.tfLiteWrapper = TfliteWrapper(modelFileName: modelFileName)
  guard self.tfLiteWrapper.setUpModelAndInterpreter() else {
    return nil
  }

  super.init()

  tfLiteWrapper.setNumberOfThreads(threadCount)

  // Opens and loads the classes listed in the labels file
  loadLabels(fromFileName: labelsFileName, fileExtension: labelsFileExtension)
}
```

#### Process input

The method `runModel` accepts a `CVPixelBuffer` of camera data, which can be
obtained from the `didOutput` method defined in `ViewController`.

We crop the image, call `CVPixelBufferLockBaseAddress` to prepare the buffer to
be read by the CPU, and then create an input tensor using the TensorFlow Lite
wrapper:

```swift
guard  let tensorInputBaseAddress = tfLiteWrapper.inputTensor(at: 0) else {
  return nil
}
```

The image buffer contains an encoded color for each pixel in `BGRA` format
(where `A` represents Alpha, or transparency), and our model expects it in `RGB`
format. We now step through the buffer four bytes at a time, copying the three
bytes we care about (`R`, `G`, and `B`) to the input tensor.

Note: Since we are using a quantized model, we can directly use the `UInt8`
values from the buffer. If we were using a float model, we would have to convert
them to floating point by dividing by 255.

```swift
let inputImageBaseAddress = sourceStartAddrss.assumingMemoryBound(to: UInt8.self)

for y in 0...wantedInputHeight - 1 {
  let tensorInputRow = tensorInputBaseAddress.advanced(by: (y * wantedInputWidth * wantedInputChannels))
  let inputImageRow = inputImageBaseAddress.advanced(by: y * wantedInputWidth * imageChannels)

  for x in 0...wantedInputWidth - 1 {

    let out_pixel = tensorInputRow.advanced(by: x * wantedInputChannels)
    let in_pixel = inputImageRow.advanced(by: x * imageChannels)

    var b = 2
    for c in 0...(wantedInputChannels) - 1 {

      // We are reversing the order of pixels since the source pixel format is BGRA, but the model requires RGB format.
      out_pixel[c] = in_pixel[b]
      b = b - 1
    }
  }
}
```

#### Run inference

Running inference is a simple call to `tfLiteWrapper.invokeInterpreter()`. The
result of this synchronous call can be obtained by calling
`tfLiteWrapper.outputTensor()`.

```swift
guard tfLiteWrapper.invokeInterpreter() else {
  return nil
}

guard let outputTensor = tfLiteWrapper.outputTensor(at: 0) else {
  return nil
}
```

#### Process results

The `getTopN` method, also declared in `ModelDataHandler.swift`, interprets the
contents of the output tensor. It returns a list of the top N predictions,
ordered by confidence.

The output tensor contains one `UInt8` value per class label, with a value
between 0 and 255 corresponding to a confidence of 0 to 100% that each label is
present in the image.

First, the results are mapped into an array of `Inference` instances, each with
a `confidence` between 0 and 1 and a `className` representing the label.

```swift
for i in 0...predictionSize - 1 {
  let value = Double(prediction[i]) / 255.0

  guard i < labels.count else {
    continue
  }

  let inference = Inference(confidence: value, className: labels[i])
  resultsArray.append(inference)
}
```

Next, the results are sorted, and we return the top `N` (where N is
`resultCount`).

```swift
resultsArray.sort { (first, second) -> Bool in
  return first.confidence  > second.confidence
}

guard resultsArray.count > resultCount else {
  return resultsArray
}
let finalArray = resultsArray[0..<resultCount]

return Array(finalArray)
```

### Display results

The file
[`InferenceViewController.swift`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/ViewControllers/InferenceViewController.swift)
defines the app's UI.

A `UITableView` instance, `tableView`, is used to display the results.
