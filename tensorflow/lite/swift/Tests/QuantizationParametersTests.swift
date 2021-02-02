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

class QuantizationParametersTests: XCTestCase {

  func testInitWithCustomValues() {
    let parameters = QuantizationParameters(scale: 0.5, zeroPoint: 1)
    XCTAssertEqual(parameters.scale, 0.5)
    XCTAssertEqual(parameters.zeroPoint, 1)
  }

  func testEquatable() {
    let parameters1 = QuantizationParameters(scale: 0.5, zeroPoint: 1)
    let parameters2 = QuantizationParameters(scale: 0.5, zeroPoint: 1)
    XCTAssertEqual(parameters1, parameters2)

    let parameters3 = QuantizationParameters(scale: 0.4, zeroPoint: 1)
    XCTAssertNotEqual(parameters1, parameters3)
    XCTAssertNotEqual(parameters2, parameters3)
  }
}
