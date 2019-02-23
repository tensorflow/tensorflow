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

class InterpreterOptionsTests: XCTestCase {

  func testInterpreterOptions_InitWithDefaultValues() {
    let options = InterpreterOptions()
    XCTAssertNil(options.threadCount)
    XCTAssertFalse(options.isErrorLoggingEnabled)
  }

  func testInterpreterOptions_InitWithCustomValues() {
    var options = InterpreterOptions()
    options.threadCount = 2
    XCTAssertEqual(options.threadCount, 2)
    options.isErrorLoggingEnabled = true
    XCTAssertTrue(options.isErrorLoggingEnabled)
  }

  func testInterpreterOptions_Equatable() {
    var options1 = InterpreterOptions()
    var options2 = InterpreterOptions()
    XCTAssertEqual(options1, options2)

    options1.threadCount = 2
    options2.threadCount = 2
    XCTAssertEqual(options1, options2)

    options2.threadCount = 3
    XCTAssertNotEqual(options1, options2)
    options2.threadCount = 2

    options1.isErrorLoggingEnabled = true
    options2.isErrorLoggingEnabled = true
    XCTAssertEqual(options1, options2)

    options2.isErrorLoggingEnabled = false
    XCTAssertNotEqual(options1, options2)
  }
}
