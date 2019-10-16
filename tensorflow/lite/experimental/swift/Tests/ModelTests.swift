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

class ModelTests: XCTestCase {

  var modelPath: String!

  override func setUp() {
    super.setUp()

    let bundle = Bundle(for: type(of: self))
    guard let modelPath = bundle.path(
      forResource: Constant.modelInfo.name,
      ofType: Constant.modelInfo.extension
    ) else {
      XCTFail("Failed to get the model file path.")
      return
    }
    self.modelPath = modelPath
  }

  override func tearDown() {
    modelPath = nil

    super.tearDown()
  }

  func testInitWithFilePath() {
    XCTAssertNotNil(Model(filePath: modelPath))
  }

  func testInitWithEmptyFilePath_FailsInitialization() {
    XCTAssertNil(Model(filePath: ""))
  }

  func testInitWithInvalidFilePath_FailsInitialization() {
    XCTAssertNil(Model(filePath: "invalid/path"))
  }
}

// MARK: - Constants

private enum Constant {
  static let modelInfo = (name: "add", extension: "bin")
}
