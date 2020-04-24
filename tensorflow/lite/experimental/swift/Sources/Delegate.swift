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

import TensorFlowLiteC

/// A delegate that the `Interpreter` uses to perform TensorFlow Lite model computations.
public protocol Delegate: class {
  /// The `TfLiteDelegate` C pointer type.
  typealias CDelegate = UnsafeMutablePointer<TfLiteDelegate>

  /// The delegate that performs model computations.
  var cDelegate: CDelegate? { get }
}
