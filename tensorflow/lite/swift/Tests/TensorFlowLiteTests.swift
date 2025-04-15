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

class TensorFlowLiteTests: XCTestCase {

  func testRuntime_Version() {
    #if swift(>=5.0)
    let pattern = #"^(\d+)\.(\d+)\.(\d+)([+-][-.0-9A-Za-z]+)?(\+\w+)?$"#
    #else
    let pattern = "^(\\d+)\\.(\\d+)\\.(\\d+)([+-][-.0-9A-Za-z]+)?(\\+\\w+)?$"
    #endif  // swift(>=5.0)
    XCTAssertNotNil(TensorFlowLite.Runtime.version.range(of: pattern, options: .regularExpression))
  }
}
