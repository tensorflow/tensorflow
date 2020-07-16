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

import XCTest

@testable import TensorFlowLite

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

  func testInit_ValidModelPath() {
    XCTAssertNoThrow(try Interpreter(modelPath: AddModel.path))
  }

  func testInit_InvalidModelPath_ThrowsFailedToLoadModel() {
    XCTAssertThrowsError(try Interpreter(modelPath: "/invalid/path")) { error in
      self.assertEqualErrors(actual: error, expected: .failedToLoadModel)
    }
  }

  func testInitWithOptions() throws {
    var options = Interpreter.Options()
    options.threadCount = 2
    let interpreter = try Interpreter(modelPath: AddQuantizedModel.path, options: options)
    XCTAssertNotNil(interpreter.options)
    XCTAssertNil(interpreter.delegates)
  }

  func testInputTensorCount() {
    XCTAssertEqual(interpreter.inputTensorCount, AddModel.inputTensorCount)
  }

  func testOutputTensorCount() {
    XCTAssertEqual(interpreter.outputTensorCount, AddModel.outputTensorCount)
  }

  func testInvoke() throws {
    try interpreter.allocateTensors()
    XCTAssertNoThrow(try interpreter.invoke())
  }

  func testInvoke_ThrowsAllocateTensorsRequired_ModelNotReady() {
    XCTAssertThrowsError(try interpreter.invoke()) { error in
      self.assertEqualErrors(actual: error, expected: .allocateTensorsRequired)
    }
  }

  func testInputTensorAtIndex() throws {
    try setUpAddModelInputTensor()
    let inputTensor = try interpreter.input(at: AddModel.validIndex)
    XCTAssertEqual(inputTensor, AddModel.inputTensor)
  }

  func testInputTensorAtIndex_QuantizedModel() throws {
    interpreter = try Interpreter(modelPath: AddQuantizedModel.path)
    try setUpAddQuantizedModelInputTensor()
    let inputTensor = try interpreter.input(at: AddQuantizedModel.inputOutputIndex)
    XCTAssertEqual(inputTensor, AddQuantizedModel.inputTensor)
  }

  func testInputTensorAtIndex_ThrowsInvalidIndex() throws {
    try interpreter.allocateTensors()
    XCTAssertThrowsError(try interpreter.input(at: AddModel.invalidIndex)) { error in
      let maxIndex = AddModel.inputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testInputTensorAtIndex_ThrowsAllocateTensorsRequired() {
    XCTAssertThrowsError(try interpreter.input(at: AddModel.validIndex)) { error in
      self.assertEqualErrors(actual: error, expected: .allocateTensorsRequired)
    }
  }

  func testOutputTensorAtIndex() throws {
    try setUpAddModelInputTensor()
    try interpreter.invoke()
    let outputTensor = try interpreter.output(at: AddModel.validIndex)
    XCTAssertEqual(outputTensor, AddModel.outputTensor)
    let expectedResults = [Float32](unsafeData: outputTensor.data)
    XCTAssertEqual(expectedResults, AddModel.results)
  }

  func testOutputTensorAtIndex_QuantizedModel() throws {
    interpreter = try Interpreter(modelPath: AddQuantizedModel.path)
    try setUpAddQuantizedModelInputTensor()
    try interpreter.invoke()
    let outputTensor = try interpreter.output(at: AddQuantizedModel.inputOutputIndex)
    XCTAssertEqual(outputTensor, AddQuantizedModel.outputTensor)
    let expectedResults = [UInt8](outputTensor.data)
    XCTAssertEqual(expectedResults, AddQuantizedModel.results)
  }

  func testOutputTensorAtIndex_ThrowsInvalidIndex() throws {
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

  func testOutputTensorAtIndex_ThrowsInvokeInterpreterRequired() {
    XCTAssertThrowsError(try interpreter.output(at: AddModel.validIndex)) { error in
      self.assertEqualErrors(actual: error, expected: .invokeInterpreterRequired)
    }
  }

  func testResizeInputTensorAtIndexToShape() {
    XCTAssertNoThrow(try interpreter.resizeInput(at: AddModel.validIndex, to: [2, 2, 3]))
    XCTAssertNoThrow(try interpreter.allocateTensors())
  }

  func testResizeInputTensorAtIndexToShape_ThrowsInvalidIndex() {
    XCTAssertThrowsError(
      try interpreter.resizeInput(
        at: AddModel.invalidIndex,
        to: [2, 2, 3]
      )
    ) { error in
      let maxIndex = AddModel.inputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testCopyDataToInputTensorAtIndex() throws {
    try interpreter.resizeInput(at: AddModel.validIndex, to: AddModel.shape)
    try interpreter.allocateTensors()
    let inputTensor = try interpreter.copy(AddModel.inputData, toInputAt: AddModel.validIndex)
    XCTAssertEqual(inputTensor.data, AddModel.inputData)
  }

  func testCopyDataToInputTensorAtIndex_ThrowsInvalidIndex() {
    XCTAssertThrowsError(
      try interpreter.copy(
        AddModel.inputData,
        toInputAt: AddModel.invalidIndex
      )
    ) { error in
      let maxIndex = AddModel.inputTensorCount - 1
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorIndex(index: AddModel.invalidIndex, maxIndex: maxIndex)
      )
    }
  }

  func testCopyDataToInputTensorAtIndex_ThrowsInvalidDataCount() throws {
    try interpreter.resizeInput(at: AddModel.validIndex, to: AddModel.shape)
    try interpreter.allocateTensors()
    let invalidData = Data(count: AddModel.dataCount - 1)
    XCTAssertThrowsError(
      try interpreter.copy(
        invalidData,
        toInputAt: AddModel.validIndex
      )
    ) { error in
      self.assertEqualErrors(
        actual: error,
        expected: .invalidTensorDataCount(provided: invalidData.count, required: AddModel.dataCount)
      )
    }
  }

  func testAllocateTensors() {
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

class InterpreterOptionsTests: XCTestCase {

  func testInitWithDefaultValues() {
    let options = Interpreter.Options()
    XCTAssertNil(options.threadCount)
    XCTAssertFalse(options.isXNNPackEnabled)
  }

  func testInitWithCustomValues() {
    var options = Interpreter.Options()

    options.threadCount = 2
    XCTAssertEqual(options.threadCount, 2)

    options.isXNNPackEnabled = false
    XCTAssertFalse(options.isXNNPackEnabled)

    options.isXNNPackEnabled = true
    XCTAssertTrue(options.isXNNPackEnabled)
  }

  func testEquatable() {
    var options1 = Interpreter.Options()
    var options2 = Interpreter.Options()
    XCTAssertEqual(options1, options2)

    options1.threadCount = 2
    options2.threadCount = 2
    XCTAssertEqual(options1, options2)

    options2.threadCount = 3
    XCTAssertNotEqual(options1, options2)

    options2.threadCount = 2
    XCTAssertEqual(options1, options2)

    options2.isXNNPackEnabled = true
    XCTAssertNotEqual(options1, options2)

    options1.isXNNPackEnabled = true
    XCTAssertEqual(options1, options2)
  }
}

// MARK: - Constants

/// Values for the `add.bin` model.
enum AddModel {
  static let info = (name: "add", extension: "bin")
  static let inputTensorCount = 1
  static let outputTensorCount = 1
  static let invalidIndex = 1
  static let validIndex = 0
  static let shape: Tensor.Shape = [2]
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
enum AddQuantizedModel {
  static let info = (name: "add_quantized", extension: "bin")
  static let inputOutputIndex = 0
  static let shape: Tensor.Shape = [2]
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
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
      self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
      self = unsafeData.withUnsafeBytes {
        .init(
          UnsafeBufferPointer<Element>(
            start: $0,
            count: unsafeData.count / MemoryLayout<Element>.stride
          ))
      }
    #endif  // swift(>=5.0)
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
