# TensorFlow Lite iOS image classification example

This document walks through the code of a simple iOS mobile application that
demonstrates [image classification](overview.md) using the device camera.

The application code is located in the
[Tensorflow examples](https://github.com/tensorflow/examples) repository, along
with instructions for building and deploying the app.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">Example
application</a>

## Explore the code

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift)
for performing image classification.

Note: Objective-C developers should use the TensorFlow Lite
[Objective-C library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc).

We're now going to walk through the most important parts of the sample code.

### Get camera input

The app's main view is represented by the `ViewController` class in
[`ViewController.swift`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/ViewControllers/ViewController.swift),
which we extend with functionality from the `CameraFeedManagerDelegate` protocol
to process frames from a camera feed. To run inference on a given frame, we
implement the `didOutput` method, which is called whenever a frame is available
from the camera.

Our implementation of `didOutput` includes a call to the `runModel` method of a
`ModelDataHandler` instance. As we will see below, this class gives us access to
the TensorFlow Lite `Interpreter` class for performing image classification.

```swift
extension ViewController: CameraFeedManagerDelegate {

  func didOutput(pixelBuffer: CVPixelBuffer) {
    let currentTimeMs = Date().timeIntervalSince1970 * 1000
    guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else { return }
    previousInferenceTimeMs = currentTimeMs

    // Pass the pixel buffer to TensorFlow Lite to perform inference.
    result = modelDataHandler?.runModel(onFrame: pixelBuffer)

    // Display results by handing off to the InferenceViewController.
    DispatchQueue.main.async {
      let resolution = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
      self.inferenceViewController?.inferenceResult = self.result
      self.inferenceViewController?.resolution = resolution
      self.inferenceViewController?.tableView.reloadData()
    }
  }
...
```

### ModelDataHandler

The Swift class `ModelDataHandler`, defined in
[`ModelDataHandler.swift`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/ModelDataHandler/ModelDataHandler.swift),
handles all data preprocessing and makes calls to run inference on a given frame
using the TensorFlow Lite [`Interpreter`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/swift/Sources/Interpreter.swift).
It then formats the inferences obtained from invoking the `Interpreter` and
returns the top N results for a successful inference.

The following sections show how this works.

#### Initialization

The `init` method creates a new instance of the `Interpreter` and loads the
specified model and labels files from the app's main bundle.

```swift
init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
  let modelFilename = modelFileInfo.name

  // Construct the path to the model file.
  guard let modelPath = Bundle.main.path(
    forResource: modelFilename,
    ofType: modelFileInfo.extension
  ) else {
    print("Failed to load the model file with name: \(modelFilename).")
    return nil
  }

  // Specify the options for the `Interpreter`.
  self.threadCount = threadCount
  var options = InterpreterOptions()
  options.threadCount = threadCount
  options.isErrorLoggingEnabled = true
  do {
    // Create the `Interpreter`.
    interpreter = try Interpreter(modelPath: modelPath, options: options)
  } catch let error {
    print("Failed to create the interpreter with error: \(error.localizedDescription)")
    return nil
  }
  // Load the classes listed in the labels file.
  loadLabels(fileInfo: labelsFileInfo)
}
```

#### Process input

The method `runModel` accepts a `CVPixelBuffer` of camera data, which can be
obtained from the `didOutput` method defined in `ViewController`.

We crop the image to the size that the model was trained on. For example,
`224x224` for the MobileNet v1 model.

The image buffer contains an encoded color for each pixel in `BGRA` format
(where `A` represents Alpha, or transparency). Our model expects the format to
be `RGB`, so we use the following helper method to remove the alpha component
from the image buffer to get the `RGB` data representation:

```swift
private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
private func rgbDataFromBuffer(
  _ buffer: CVPixelBuffer,
  byteCount: Int,
  isModelQuantized: Bool
) -> Data? {
  CVPixelBufferLockBaseAddress(buffer, .readOnly)
  defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
  guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
    return nil
  }
  let count = CVPixelBufferGetDataSize(buffer)
  let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
  var rgbBytes = [UInt8](repeating: 0, count: byteCount)
  var index = 0
  for component in bufferData.enumerated() {
    let offset = component.offset
    let isAlphaComponent = (offset % alphaComponent.baseOffset) == alphaComponent.moduloRemainder
    guard !isAlphaComponent else { continue }
    rgbBytes[index] = component.element
    index += 1
  }
  if isModelQuantized { return Data(bytes: rgbBytes) }
  return Data(copyingBufferOf: rgbBytes.map { Float($0) / 255.0 })
}
```

#### Run inference

Here's the code for getting the `RGB` data representation of the pixel buffer,
copying that data to the input
[`Tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/swift/Sources/Tensor.swift),
and running inference by invoking the `Interpreter`:

```swift
let outputTensor: Tensor
do {
  // Allocate memory for the model's input `Tensor`s.
  try interpreter.allocateTensors()
  let inputTensor = try interpreter.input(at: 0)

  // Remove the alpha component from the image buffer to get the RGB data.
  guard let rgbData = rgbDataFromBuffer(
    thumbnailPixelBuffer,
    byteCount: batchSize * inputWidth * inputHeight * inputChannels,
    isModelQuantized: inputTensor.dataType == .uInt8
  ) else {
    print("Failed to convert the image buffer to RGB data.")
    return
  }

  // Copy the RGB data to the input `Tensor`.
  try interpreter.copy(rgbData, toInputAt: 0)

  // Run inference by invoking the `Interpreter`.
  try interpreter.invoke()

  // Get the output `Tensor` to process the inference results.
  outputTensor = try interpreter.output(at: 0)
} catch let error {
  print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
  return
}
```

#### Process results

If the model is quantized, the output `Tensor` contains one `UInt8` value per
class label. Dequantize the results so the values are floats, ranging from 0.0
to 1.0, where each value represents the confidence that a label is present in
the image:

```swift
guard let quantization = outputTensor.quantizationParameters else {
  print("No results returned because the quantization values for the output tensor are nil.")
  return
}

// Get the quantized results from the output tensor's `data` property.
let quantizedResults = [UInt8](outputTensor.data)

// Dequantize the results using the quantization values.
let results = quantizedResults.map {
  quantization.scale * Float(Int($0) - quantization.zeroPoint)
}
```

Next, the results are sorted to get the top `N` results (where `N` is
`resultCount`):

```swift
// Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
let zippedResults = zip(labels.indices, results)

// Sort the zipped results by confidence value in descending order.
let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)

// Get the top N `Inference` results.
let topNInferences = sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
```

### Display results

The file
[`InferenceViewController.swift`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/ImageClassification/ViewControllers/InferenceViewController.swift)
defines the app's UI. A `UITableView` is used to display the results.
