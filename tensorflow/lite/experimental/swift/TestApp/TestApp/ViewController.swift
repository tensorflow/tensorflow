// Copyright 2019 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlowLite
import UIKit

class ViewController: UIViewController {

  // MARK: - Properties

  /// TensorFlow Lite interpreter object for performing inference from a given model.
  private var interpreter: Interpreter?

  /// Serial dispatch queue for managing `Interpreter` calls.
  private let interpreterQueue = DispatchQueue(
    label: Constant.dispatchQueueLabel,
    qos: .userInitiated
  )

  /// The currently selected model.
  private var currentModel: Model {
    guard let currentModel = Model(rawValue: modelControl.selectedSegmentIndex) else {
      preconditionFailure("Invalid model for selected segment index.")
    }
    return currentModel
  }

  /// A description of the current model.
  private var modelDescription: String {
    guard let interpreter = interpreter else { return "" }
    let inputCount = interpreter.inputTensorCount
    let outputCount = interpreter.outputTensorCount
    let inputTensors = (0..<inputCount).map { index in
      var tensorInfo = "  Input \(index + 1): "
      do {
        let tensor = try interpreter.input(at: index)
        tensorInfo += "\(tensor)"
      } catch let error {
        tensorInfo += "\(error.localizedDescription)"
      }
      return tensorInfo
    }.joined(separator: "\n")
    let outputTensors = (0..<outputCount).map { index in
      var tensorInfo = "  Output \(index + 1): "
      do {
        let tensor = try interpreter.output(at: index)
        tensorInfo += "\(tensor)"
      } catch let error {
        tensorInfo += "\(error.localizedDescription)"
      }
      return tensorInfo
    }.joined(separator: "\n")
    return "Model Description:\n" +
             "  Input Tensor Count = \(inputCount)\n\(inputTensors)\n\n" +
             "  Output Tensor Count = \(outputCount)\n\(outputTensors)"
  }

  // MARK: - IBOutlets

  /// A segmented control for changing models. See the `Model` enum for available models.
  @IBOutlet private var modelControl: UISegmentedControl!

  @IBOutlet private var resultsTextView: UITextView!
  @IBOutlet private var invokeButton: UIBarButtonItem!

  // MARK: - UIViewController

  override func viewDidLoad() {
    super.viewDidLoad()

    invokeButton.isEnabled = false
    updateResultsText("Using TensorFlow Lite runtime version \(TensorFlowLite.Runtime.version).")
    loadModel()
  }

  // MARK: - IBActions

  @IBAction func modelChanged(_ sender: Any) {
    invokeButton.isEnabled = false
    updateResultsText("Switched to the \(currentModel.description) model.")
    loadModel()
  }

  @IBAction func invokeInterpreter(_ sender: Any) {
    switch currentModel {
    case .add:
      invokeAdd()
    case .addQuantized:
      invokeAddQuantized()
    case .multiAdd:
      invokeMultiAdd()
    }
  }

  // MARK: - Private

  private func loadModel() {
    let fileInfo = currentModel.fileInfo
    guard let modelPath = Bundle.main.path(forResource: fileInfo.name, ofType: fileInfo.extension)
    else {
      updateResultsText("Failed to load the \(currentModel.description) model.")
      return
    }
    setUpInterpreter(withModelPath: modelPath)
  }

  private func setUpInterpreter(withModelPath modelPath: String) {
    interpreterQueue.async {
      do {
        var options = Interpreter.Options()
        options.threadCount = 2
        self.interpreter = try Interpreter(modelPath: modelPath, options: options)
      } catch let error {
        self.updateResultsText(
          "Failed to create the interpreter with error: \(error.localizedDescription)"
        )
        return
      }
      safeDispatchOnMain { self.invokeButton.isEnabled = true }
    }
  }

  private func invokeAdd() {
    interpreterQueue.async {
      guard let interpreter = self.interpreter else {
        self.updateResultsText(Constant.nilInterpreterErrorMessage)
        return
      }
      do {
        try interpreter.resizeInput(at: 0, to: [2])
        try interpreter.allocateTensors()
        let input: [Float32] = [1, 3]
        let resultsText = self.modelDescription + "\n\n" +
          "Performing 2 add operations on input \(input.description) equals: "
        self.updateResultsText(resultsText)
        let data = Data(copyingBufferOf: input)
        try interpreter.copy(data, toInputAt: 0)
        try interpreter.invoke()
        let outputTensor = try interpreter.output(at: 0)
        let results: () -> String = {
          guard let results = [Float32](unsafeData: outputTensor.data) else { return "No results." }
          return resultsText + results.description
        }
        self.updateResultsText(results())
      } catch let error {
        self.updateResultsText(
          "Failed to invoke the interpreter with error: \(error.localizedDescription)"
        )
        return
      }
    }
  }

  private func invokeAddQuantized() {
    interpreterQueue.async {
      guard let interpreter = self.interpreter else {
        self.updateResultsText(Constant.nilInterpreterErrorMessage)
        return
      }
      do {
        try interpreter.resizeInput(at: 0, to: [2])
        try interpreter.allocateTensors()
        let input: [UInt8] = [1, 3]
        let resultsText = self.modelDescription + "\n\n" +
          "Performing 2 add operations on quantized input \(input.description) equals: "
        self.updateResultsText(resultsText)
        let data = Data(input)
        try interpreter.copy(data, toInputAt: 0)
        try interpreter.invoke()
        let outputTensor = try interpreter.output(at: 0)
        let results: () -> String = {
          guard let quantizationParameters = outputTensor.quantizationParameters else {
            return "No results."
          }
          let quantizedResults = [UInt8](outputTensor.data)
          let dequantizedResults = quantizedResults.map {
            quantizationParameters.scale * Float(Int($0) - quantizationParameters.zeroPoint)
          }
          return resultsText + quantizedResults.description +
                   ", dequantized results: " + dequantizedResults.description
        }
        self.updateResultsText(results())
      } catch let error {
        self.updateResultsText(
          "Failed to invoke the interpreter with error: \(error.localizedDescription)"
        )
        return
      }
    }
  }

  private func invokeMultiAdd() {
    interpreterQueue.async {
      guard let interpreter = self.interpreter else {
        self.updateResultsText(Constant.nilInterpreterErrorMessage)
        return
      }
      do {
        let shape = Tensor.Shape(2)
        try (0..<interpreter.inputTensorCount).forEach { index in
          try interpreter.resizeInput(at: index, to: shape)
        }
        try interpreter.allocateTensors()
        let inputs = try (0..<interpreter.inputTensorCount).map { index -> [Float32] in
          let input = [Float32(index + 1), Float32(index + 2)]
          let data = Data(copyingBufferOf: input)
          try interpreter.copy(data, toInputAt: index)
          return input
        }
        let resultsText = self.modelDescription + "\n\n" +
          "Performing 3 add operations on inputs \(inputs.description) equals: "
        self.updateResultsText(resultsText)
        try interpreter.invoke()
        let results = try (0..<interpreter.outputTensorCount).map { index -> [Float32] in
          let tensor = try interpreter.output(at: index)
          return [Float32](unsafeData: tensor.data) ?? []
        }
        self.updateResultsText(resultsText + results.description)
      } catch let error {
        self.updateResultsText(
          "Failed to invoke the interpreter with error: \(error.localizedDescription)"
        )
        return
      }
    }
  }

  private func updateResultsText(_ text: String? = nil) {
    safeDispatchOnMain { self.resultsTextView.text = text }
  }
}

// MARK: - Constants

private enum Constant {
  static let dispatchQueueLabel = "TensorFlowLiteInterpreterQueue"
  static let nilInterpreterErrorMessage =
    "Failed to invoke the interpreter because the interpreter was nil."
}

/// Models that can be loaded by the TensorFlow Lite `Interpreter`.
private enum Model: Int, CustomStringConvertible {
  /// A float model that performs two add operations on one input tensor and returns the result in
  /// one output tensor.
  case add = 0
  /// A quantized model that performs two add operations on one input tensor and returns the result
  /// in one output tensor.
  case addQuantized = 1
  /// A float model that performs three add operations on four input tensors and returns the results
  /// in 2 output tensors.
  case multiAdd = 2

  var fileInfo: (name: String, extension: String) {
    switch self {
    case .add:
      return Add.fileInfo
    case .addQuantized:
      return AddQuantized.fileInfo
    case .multiAdd:
      return MultiAdd.fileInfo
    }
  }

  // MARK: - CustomStringConvertible

  var description: String {
    switch self {
    case .add:
      return Add.name
    case .addQuantized:
      return AddQuantized.name
    case .multiAdd:
      return MultiAdd.name
    }
  }
}

/// Values for the `Add` model.
private enum Add {
  static let name = "Add"
  static let fileInfo = (name: "add", extension: "bin")
}

/// Values for the `AddQuantized` model.
private enum AddQuantized {
  static let name = "AddQuantized"
  static let fileInfo = (name: "add_quantized", extension: "bin")
}

/// Values for the `MultiAdd` model.
private enum MultiAdd {
  static let name = "MultiAdd"
  static let fileInfo = (name: "multi_add", extension: "bin")
}

// MARK: - Fileprivate

/// Safely dispatches the given block on the main queue. If the current thread is `main`, the block
/// is executed synchronously; otherwise, the block is executed asynchronously on the main thread.
fileprivate func safeDispatchOnMain(_ block: @escaping () -> Void) {
  if Thread.isMainThread { block(); return }
  DispatchQueue.main.async { block() }
}
