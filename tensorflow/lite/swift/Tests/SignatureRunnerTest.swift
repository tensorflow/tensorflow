// Copyright 2022 Google Inc. All rights reserved.
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

class SignatureRunnerTest: XCTestCase {

  func testSignatureKeys() throws {
    let interpreter = try Interpreter(modelPath: MultiSignaturesModel.path)
    XCTAssertEqual(interpreter.signatureKeys, MultiSignaturesModel.signatureKeys)
    XCTAssertNotNil(try interpreter.signatureRunner(with: MultiSignaturesModel.AddSignature.key))
    XCTAssertThrowsError(try interpreter.signatureRunner(with: "dummy")) { error in
      self.assertEqualErrors(
        actual: error, expected: .failedToCreateSignatureRunner(signatureKey: "dummy"))
    }
  }

  func testResizeInputTensor() throws {
    let interpreter = try Interpreter(modelPath: MultiSignaturesModel.path)
    let addRunner = try interpreter.signatureRunner(with: MultiSignaturesModel.AddSignature.key)
    XCTAssertEqual(addRunner.inputs, MultiSignaturesModel.AddSignature.inputs)
    let inputTensor = try addRunner.input(named: "x")
    // Validate signature "add" input tensor "x" before resizing.
    XCTAssertEqual(inputTensor.name, MultiSignaturesModel.AddSignature.inputTensor.name)
    XCTAssertEqual(inputTensor.dataType, MultiSignaturesModel.AddSignature.inputTensor.dataType)
    XCTAssertEqual(inputTensor.shape, [1])
    // Test fail to copy data before resizing the tensor
    XCTAssertThrowsError(
      try addRunner.copy(MultiSignaturesModel.AddSignature.inputData, toInputNamed: "x")
    ) { error in
      self.assertEqualErrors(
        actual: error, expected: .invalidTensorDataCount(provided: 8, required: 4))
    }
    // Resize signature "add" input tensor "x"
    try addRunner.resizeInput(named: "x", toShape: MultiSignaturesModel.AddSignature.shape)
    try addRunner.allocateTensors()
    // Copy data to input tensor "x"
    try addRunner.copy(MultiSignaturesModel.AddSignature.inputData, toInputNamed: "x")
    // Validate signature "add" input tensor "x" after resizing and copying data.
    XCTAssertEqual(
      try addRunner.input(named: "x"), MultiSignaturesModel.AddSignature.inputTensor)
  }

  func testResizeInputTensor_invalidTensor() throws {
    let interpreter = try Interpreter(modelPath: MultiSignaturesModel.path)
    let addRunner = try interpreter.signatureRunner(with: MultiSignaturesModel.AddSignature.key)
    // Test fail to get input tensor for a dummy input name.
    XCTAssertThrowsError(
      try addRunner.input(named: "dummy")
    ) { error in
      self.assertEqualErrors(
        actual: error, expected: .failedToGetTensor(tensorType: "input", nameInSignature: "dummy"))
    }
    // Test fail to resize dummy input tensor
    XCTAssertThrowsError(
      try addRunner.resizeInput(named: "dummy", toShape: [2])
    ) { error in
      self.assertEqualErrors(
        actual: error, expected: .failedToResizeInputTensor(inputName: "dummy"))
    }
  }

  func testInvokeWithInputs() throws {
    let interpreter = try Interpreter(modelPath: MultiSignaturesModel.path)
    let addRunner = try interpreter.signatureRunner(with: MultiSignaturesModel.AddSignature.key)
    XCTAssertEqual(addRunner.outputs, MultiSignaturesModel.AddSignature.outputs)
    // Validate signature "add" output tensor "output_0" before inference
    let outputTensor = try addRunner.output(named: "output_0")
    XCTAssertEqual(outputTensor.name, MultiSignaturesModel.AddSignature.outputTensor.name)
    XCTAssertEqual(outputTensor.dataType, MultiSignaturesModel.AddSignature.outputTensor.dataType)
    XCTAssertEqual(outputTensor.shape, [1])
    // Resize signature "add" input tensor "x"
    try addRunner.resizeInput(named: "x", toShape: MultiSignaturesModel.AddSignature.shape)
    // Invoke signature "add" with inputs.
    try addRunner.invoke(with: ["x": MultiSignaturesModel.AddSignature.inputData])
    // Validate signature "add" output tensor "output_0" after inference
    XCTAssertEqual(
      try addRunner.output(named: "output_0"), MultiSignaturesModel.AddSignature.outputTensor)
  }

  // MARK: - Private

  private func assertEqualErrors(actual: Error, expected: SignatureRunnerError) {
    guard let actual = actual as? SignatureRunnerError else {
      XCTFail("Actual error should be of type SignatureRunnerError.")
      return
    }
    XCTAssertEqual(actual, expected)
  }
}

// MARK: - Constants

/// Values for the `multi_signatures.bin` model.
enum MultiSignaturesModel {
  static let info = (name: "multi_signatures", extension: "bin")
  static let signatureKeys = [AddSignature.key, SubSignature.key]
  static var path: String = {
    let bundle = Bundle(for: SignatureRunnerTest.self)
    guard let path = bundle.path(forResource: info.name, ofType: info.extension) else { return "" }
    return path
  }()

  enum AddSignature {
    static let key = "add"
    static let inputs = ["x"]
    static let outputs = ["output_0"]
    static let inputData = Data(copyingBufferOf: [Float32(2.0), Float32(4.0)])
    static let outputData = Data(copyingBufferOf: [Float32(4.0), Float32(6.0)])
    static let shape: Tensor.Shape = [2]
    static let inputTensor = Tensor(
      name: "add_x:0",
      dataType: .float32,
      shape: shape,
      data: inputData
    )
    static let outputTensor = Tensor(
      name: "StatefulPartitionedCall:0",
      dataType: .float32,
      shape: shape,
      data: outputData
    )
  }

  enum SubSignature {
    static let key = "sub"
  }
}
