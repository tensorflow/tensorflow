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

import XCTest

@testable import TensorFlowLite

class MetalDelegateTests: XCTestCase {

  func testInitDefaultGPUDelegateOptions() {
    let delegate = MetalDelegate()
    XCTAssertFalse(delegate.options.isPrecisionLossAllowed)
    XCTAssertEqual(delegate.options.waitType, .passive)
  }

  func testInitWithCustomGPUDelegateOptions() {
    var options = MetalDelegate.Options()
    options.isPrecisionLossAllowed = true
    options.waitType = .active
    let delegate = MetalDelegate(options: options)
    XCTAssertTrue(delegate.options.isPrecisionLossAllowed)
    XCTAssertEqual(delegate.options.waitType, .active)
  }

  func testInitInterpreterWithDelegate() throws {
    // If metal device is not available, skip.
    if MTLCreateSystemDefaultDevice() == nil {
      return
    }
    let metalDelegate = MetalDelegate()
    let interpreter = try Interpreter(modelPath: MultiAddModel.path, delegates: [metalDelegate])
    XCTAssertEqual(interpreter.delegates?.count, 1)
    XCTAssertNil(interpreter.options)
  }

  func testInitInterpreterWithOptionsAndDelegate() throws {
    // If metal device is not available, skip.
    if MTLCreateSystemDefaultDevice() == nil {
      return
    }
    var options = Interpreter.Options()
    options.threadCount = 1
    let metalDelegate = MetalDelegate()
    let interpreter = try Interpreter(
      modelPath: MultiAddModel.path,
      options: options,
      delegates: [metalDelegate]
    )
    XCTAssertNotNil(interpreter.options)
    XCTAssertEqual(interpreter.delegates?.count, 1)
  }
}

class MetalDelegateOptionsTests: XCTestCase {

  func testInitWithDefaultValues() {
    let options = MetalDelegate.Options()
    XCTAssertFalse(options.isPrecisionLossAllowed)
    XCTAssertEqual(options.waitType, .passive)
  }

  func testInitWithCustomValues() {
    var options = MetalDelegate.Options()
    options.isPrecisionLossAllowed = true
    options.waitType = .active
    XCTAssertTrue(options.isPrecisionLossAllowed)
    XCTAssertEqual(options.waitType, .active)
  }

  func testEquatable() {
    var options1 = MetalDelegate.Options()
    var options2 = MetalDelegate.Options()
    XCTAssertEqual(options1, options2)

    options1.isPrecisionLossAllowed = true
    options2.isPrecisionLossAllowed = true
    XCTAssertEqual(options1, options2)

    options1.waitType = .none
    options2.waitType = .none
    XCTAssertEqual(options1, options2)

    options2.isPrecisionLossAllowed = false
    XCTAssertNotEqual(options1, options2)
    options1.isPrecisionLossAllowed = false

    options1.waitType = .aggressive
    XCTAssertNotEqual(options1, options2)
  }
}


/// Values for the `multi_add.bin` model.
enum MultiAddModel {
  static let info = (name: "multi_add", extension: "bin")

  static var path: String = {
    let bundle = Bundle(for: MetalDelegateTests.self)
    guard let path = bundle.path(forResource: info.name, ofType: info.extension) else { return "" }
    return path
  }()
}

