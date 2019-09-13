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

/// A delegate that uses the `Metal` framework for performing TensorFlow Lite graph operations with
/// GPU acceleration.
///
/// - Important: This is an experimental interface that is subject to change.
public final class MetalDelegate: Delegate {
  /// The configuration options for the `MetalDelegate`.
  public let options: Options

  // Conformance to the `Delegate` protocol.
  public private(set) var cDelegate: CDelegate

  /// Creates a new instance configured with the given `options`.
  ///
  /// - Parameters:
  ///   - options: Configurations for the delegate. The default is a new instance of
  ///       `MetalDelegate.Options` with the default configuration values.
  public init(options: Options = Options()) {
    self.options = options
    var delegateOptions = TFLGpuDelegateOptions()
    delegateOptions.allow_precision_loss = options.allowsPrecisionLoss
    delegateOptions.wait_type = options.waitType.cWaitType
    cDelegate = TFLGpuDelegateCreate(&delegateOptions)
  }

  deinit {
    TFLGpuDelegateDelete(cDelegate)
  }
}

extension MetalDelegate {
  /// Options for configuring the `MetalDelegate`.
  public struct Options: Equatable, Hashable {
    /// Indicates whether the GPU delegate allows precision loss, such as allowing `Float16`
    /// precision for a `Float32` computation. The default is `false`.
    public var allowsPrecisionLoss = false

    /// A type indicating how the current thread should wait for work on the GPU to complete. The
    /// default is `passive`.
    public var waitType: ThreadWaitType = .passive

    /// Creates a new instance with the default values.
    public init() {}
  }
}

/// A type indicating how the current thread should wait for work scheduled on the GPU to complete.
public enum ThreadWaitType: Equatable, Hashable {
  /// The thread does not wait for the work to complete. Useful when the output of the work is used
  /// with the GPU pipeline.
  case none
  /// The thread waits until the work is complete.
  case passive
  /// The thread waits for the work to complete with minimal latency, which may require additional
  /// CPU resources.
  case active
  /// The thread waits for the work while trying to prevent the GPU from going into sleep mode.
  case aggressive

  /// The C `TFLGpuDelegateWaitType` for the current `ThreadWaitType`.
  var cWaitType: TFLGpuDelegateWaitType {
    switch self {
    case .none:
      return TFLGpuDelegateWaitTypeDoNotWait
    case .passive:
      return TFLGpuDelegateWaitTypePassive
    case .active:
      return TFLGpuDelegateWaitTypeActive
    case .aggressive:
      return TFLGpuDelegateWaitTypeAggressive
    }
  }
}
