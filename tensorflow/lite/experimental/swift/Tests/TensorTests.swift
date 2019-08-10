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

class TensorTests: XCTestCase {

  // MARK: - Tensor

  func testTensor_Init() {
    let name = "InputTensor"
    let dataType: TensorDataType = .uInt8
    let shape = TensorShape(Constant.dimensions)
    guard let data = name.data(using: .utf8) else { XCTFail("Data should not be nil."); return }
    let quantizationParameters = QuantizationParameters(scale: 0.5, zeroPoint: 1)
    let inputTensor = Tensor(
      name: name,
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    XCTAssertEqual(inputTensor.name, name)
    XCTAssertEqual(inputTensor.dataType, dataType)
    XCTAssertEqual(inputTensor.shape, shape)
    XCTAssertEqual(inputTensor.data, data)
    XCTAssertEqual(inputTensor.quantizationParameters, quantizationParameters)
  }

  func testTensor_Equatable() {
    let name = "Tensor"
    let dataType: TensorDataType = .uInt8
    let shape = TensorShape(Constant.dimensions)
    guard let data = name.data(using: .utf8) else { XCTFail("Data should not be nil."); return }
    let quantizationParameters = QuantizationParameters(scale: 0.5, zeroPoint: 1)
    let tensor1 = Tensor(
      name: name,
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    var tensor2 = Tensor(
      name: name,
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    XCTAssertEqual(tensor1, tensor2)

    tensor2 = Tensor(
      name: "Tensor2",
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    XCTAssertNotEqual(tensor1, tensor2)
  }

  // MARK: - TensorShape

  func testTensorShape_InitWithArray() {
    let shape = TensorShape(Constant.dimensions)
    XCTAssertEqual(shape.rank, Constant.dimensions.count)
    XCTAssertEqual(shape.dimensions, Constant.dimensions)
  }

  func testTensorShape_InitWithElements() {
    let shape = TensorShape(2, 2, 3)
    XCTAssertEqual(shape.rank, Constant.dimensions.count)
    XCTAssertEqual(shape.dimensions, Constant.dimensions)
  }

  func testTensorShape_InitWithArrayLiteral() {
    let shape: TensorShape = [2, 2, 3]
    XCTAssertEqual(shape.rank, Constant.dimensions.count)
    XCTAssertEqual(shape.dimensions, Constant.dimensions)
  }

  func testTensorShape_Equatable() {
    let shape1 = TensorShape(2, 2, 3)
    var shape2: TensorShape = [2, 2, 3]
    XCTAssertEqual(shape1, shape2)

    shape2 = [2, 2, 4]
    XCTAssertNotEqual(shape1, shape2)
  }
}

// MARK: - Constants

private enum Constant {
  /// Array of 2 arrays of 2 arrays of 3 numbers: [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]].
  static let dimensions = [2, 2, 3]
}
