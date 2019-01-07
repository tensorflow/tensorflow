// Copyright 2018 Google Inc. All rights reserved.
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

@testable import TensorFlowLite
import XCTest

class InterpreterTests: XCTestCase {

  var interpreter: Interpreter!

  override func setUp() {
    super.setUp()

    interpreter = try! Interpreter(modelPath: AddModel.path)
  }

  override func tearDown() {
    interpreter = nil

    super.tearDown()
  }

  func testInterpreter_InitWithModelPath() {
    XCTAssertNoThrow(try Interpreter(modelPath: AddModel.path))
  }

  func testInterpreter_Init_ThrowsFailedToLoadModel() {
    XCTAssertThrowsError(try Interpreter(modelPath: "/invalid/path")) { error in
      self.assertEqualErrors(actual: error, expected: .failedToLoadModel)
    }
  }

  func testInterpreter_InitWithModelPathAndOptions() {
    var options = InterpreterOptions()
    options.threadCount = 2
    XCTAssertNoThrow(try Interpreter(modelPath: AddModel.path, options: options))
  }

  func testInterpreter_InputTensorCount() {
    XCTAssertEqual(interpreter.inputTensorCount, AddModel.inputTensorCount)
  }

  func testInterpreter_OutputTensorCount() {
    XCTAssertEqual(interpreter.outputTensorCount, AddModel.outputTensorCount)
  }

  func testInterpreter_Invoke() throws {
    try interpreter.allocateTensors()
    XCTAssertNoThrow(try interpreter.invoke())
  }

  func testInterpreter_Invoke_ThrowsAllocateTensorsRequired_ModelNotReady() {
    XCTAssertThrowsError(try interpreter.invoke()) { error in
      self.assertEqualErrors(actual: error, expected: .allocateTensorsRequired)
    }
  }

  func testInterpreter_InputTensorAtIndex() throws {
    try setUpAddModelInputTensor()
    let inputTensor = try interpreter.input(at: AddModel.validIndex)
    XCTAssertEqual(inputTensor, AddModel.inputTensor)
  }

  func testInterpreter_InputTensorAtIndex_QuantizedModel() throws {
    interpreter = try Interpreter(modelPath: AddQuantizedModel.path)
    try setUpAddQuantizedModelInputTensor()
    let inputTensor = try interpreter.input(at: AddQuantizedModel.inputOutputIndex)
    XCTAssertEqual(inputTensor, AddQuantizedModel.inputTensor)
  }

  func testInterpreter_InputTensorAtIndex_ThrowsInvalidIndex() throws {
    try interpreter.allocateTensors()
    XCTAssertThrowsError(try interpreter.input(at: AddModel.invalidIndex)) { error in
      let maxIndex = AddModel.inputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testInterpreter_InputTensorAtIndex_ThrowsAllocateTensorsRequired() {
    XCTAssertThrowsError(try interpreter.input(at: AddModel.validIndex)) { error in
      self.assertEqualErrors(actual: error, expected: .allocateTensorsRequired)
    }
  }

  func testInterpreter_OutputTensorAtIndex() throws {
    try setUpAddModelInputTensor()
    try interpreter.invoke()
    let outputTensor = try interpreter.output(at: AddModel.validIndex)
    XCTAssertEqual(outputTensor, AddModel.outputTensor)
    let expectedResults = [Float32](unsafeData: outputTensor.data)
    XCTAssertEqual(expectedResults, AddModel.results)
  }

  func testInterpreter_OutputTensorAtIndex_QuantizedModel() throws {
    interpreter = try Interpreter(modelPath: AddQuantizedModel.path)
    try setUpAddQuantizedModelInputTensor()
    try interpreter.invoke()
    let outputTensor = try interpreter.output(at: AddQuantizedModel.inputOutputIndex)
    XCTAssertEqual(outputTensor, AddQuantizedModel.outputTensor)
    let expectedResults = [UInt8](outputTensor.data)
    XCTAssertEqual(expectedResults, AddQuantizedModel.results)
  }

  func testInterpreter_OutputTensorAtIndex_ThrowsInvalidIndex() throws {
    try interpreter.allocateTensors()
    try interpreter.invoke()
    XCTAssertThrowsError(try interpreter.output(at: AddModel.invalidIndex)) { error in
      let maxIndex = AddModel.outputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testInterpreter_OutputTensorAtIndex_ThrowsInvokeInterpreterRequired() {
    XCTAssertThrowsError(try interpreter.output(at: AddModel.validIndex)) { error in
      self.assertEqualErrors(actual: error, expected: .invokeInterpreterRequired)
    }
  }

  func testInterpreter_ResizeInputTensorAtIndexToShape() {
    XCTAssertNoThrow(try interpreter.resizeInput(at: AddModel.validIndex, to: [2, 2, 3]))
    XCTAssertNoThrow(try interpreter.allocateTensors())
  }

  func testInterpreter_ResizeInputTensorAtIndexToShape_ThrowsInvalidIndex() {
    XCTAssertThrowsError(try interpreter.resizeInput(
      at: AddModel.invalidIndex,
      to: [2, 2, 3]
    )) { error in
      let maxIndex = AddModel.inputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testInterpreter_CopyDataToInputTensorAtIndex() throws {
    try interpreter.resizeInput(at: AddModel.validIndex, to: AddModel.shape)
    try interpreter.allocateTensors()
    let inputTensor = try interpreter.copy(AddModel.inputData, toInputAt: AddModel.validIndex)
    XCTAssertEqual(inputTensor.data, AddModel.inputData)
  }

  func testInterpreter_CopyDataToInputTensorAtIndex_ThrowsInvalidIndex() {
    XCTAssertThrowsError(try interpreter.copy(
      AddModel.inputData,
      toInputAt: AddModel.invalidIndex
    )) { error in
      let maxIndex = AddModel.inputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testInterpreter_CopyDataToInputTensorAtIndex_ThrowsInvalidDataCount() throws {
    try interpreter.resizeInput(at: AddModel.validIndex, to: AddModel.shape)
    try interpreter.allocateTensors()
    let invalidData = Data(count: AddModel.dataCount - 1)
    XCTAssertThrowsError(try interpreter.copy(
      invalidData,
      toInputAt: AddModel.validIndex
    )) { error in
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorDataCount(provided: invalidData.count, required: AddModel.dataCount)
      )
    }
  }

  func testInterpreter_AllocateTensors() {
    XCTAssertNoThrow(try interpreter.allocateTensors())
  }

  // MARK: - Private

  private func setUpAddModelInputTensor() throws {
    precondition(interpreter != nil)
    try interpreter.resizeInput(at: AddModel.validIndex, to: AddModel.shape)
    try interpreter.allocateTensors()
    try interpreter.copy(AddModel.inputData, toInputAt: AddModel.validIndex)
  }

  private func setUpAddQuantizedModelInputTensor() throws {
    precondition(interpreter != nil)
    try interpreter.resizeInput(at: AddQuantizedModel.inputOutputIndex, to: AddQuantizedModel.shape)
    try interpreter.allocateTensors()
    try interpreter.copy(AddQuantizedModel.inputData, toInputAt: AddQuantizedModel.inputOutputIndex)
  }

  private func assertEqualErrors(actual: Error, expected: InterpreterError) {
    guard let actual = actual as? InterpreterError else {
      XCTFail("Actual error should be of type InterpreterError.")
      return
    }
    XCTAssertEqual(actual, expected)
  }
}

// MARK: - Constants

/// Values for the `add.bin` model.
private enum AddModel {
  static let info = (name: "add", extension: "bin")
  static let inputTensorCount = 1
  static let outputTensorCount = 1
  static let invalidIndex = 1
  static let validIndex = 0
  static let shape: TensorShape = [2]
  static let dataCount = inputData.count
  static let inputData = Data(copyingBufferOf: [Float32(1.0), Float32(3.0)])
  static let outputData = Data(copyingBufferOf: [Float32(3.0), Float32(9.0)])
  static let results = [Float32(3.0), Float32(9.0)]

  static let inputTensor = Tensor(
    name: "input",
    dataType: .float32,
    shape: shape,
    data: inputData
  )
  static let outputTensor = Tensor(
    name: "output",
    dataType: .float32,
    shape: shape,
    data: outputData
  )

  static var path: String = {
    let bundle = Bundle(for: InterpreterTests.self)
    guard let path = bundle.path(forResource: info.name, ofType: info.extension) else { return "" }
    return path
  }()
}

/// Values for the `add_quantized.bin` model.
private enum AddQuantizedModel {
  static let info = (name: "add_quantized", extension: "bin")
  static let inputOutputIndex = 0
  static let shape: TensorShape = [2]
  static let inputData = Data([1, 3])
  static let outputData = Data([3, 9])
  static let quantizationParameters = QuantizationParameters(scale: 0.003922, zeroPoint: 0)
  static let results: [UInt8] = [3, 9]

  static let inputTensor = Tensor(
    name: "input",
    dataType: .uInt8,
    shape: shape,
    data: inputData,
    quantizationParameters: quantizationParameters
  )
  static let outputTensor = Tensor(
    name: "output",
    dataType: .uInt8,
    shape: shape,
    data: outputData,
    quantizationParameters: quantizationParameters
  )

  static var path: String = {
    let bundle = Bundle(for: InterpreterTests.self)
    guard let path = bundle.path(forResource: info.name, ofType: info.extension) else { return "" }
    return path
  }()
}

// MARK: - Extensions

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    let elements = unsafeData.withUnsafeBytes {
      UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      )
    }
    self.init(elements)
  }
}

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}
