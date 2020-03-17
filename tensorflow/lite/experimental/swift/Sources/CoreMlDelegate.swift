// Copyright 2020 Google Inc. All rights reserved.
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

/// A delegate that uses the `Core ML` framework for performing TensorFlow Lite graph operations.
///
/// - Important: This is an experimental interface that is subject to change.
public final class CoreMLDelegate: Delegate {
  /// The configuration options for the `CoreMLDelegate`.
  public let options: Options

  // Conformance to the `Delegate` protocol.
  public private(set) var cDelegate: CDelegate

  /// Creates a new instance configured with the given `options`.
  ///
  /// - Parameters:
  ///   - options: Configurations for the delegate. The default is a new instance of
  ///       `CoreMLDelegate.Options` with the default configuration values.
  public init(options: Options = Options()) {
    self.options = options
    var delegateOptions = TfLiteCoreMlDelegateOptions()
    cDelegate = TfLiteCoreMlDelegateCreate(&delegateOptions)
  }

  deinit {
    TfLiteCoreMlDelegateDelete(cDelegate)
  }
}

extension CoreMLDelegate {
  /// Options for configuring the `CoreMLDelegate`.
  // TODO(b/143931022): Add preferred device support.
  public struct Options: Equatable, Hashable {
    /// Creates a new instance with the default values.
    public init() {}
  }
}
