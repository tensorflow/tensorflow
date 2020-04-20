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
  public private(set) var cDelegate: CDelegate?

  /// Creates a new instance configured with the given `options`.
  ///
  /// - Parameters:
  ///   - options: Configurations for the delegate. The default is a new instance of
  ///       `CoreMLDelegate.Options` with the default configuration values.
  public init?(options: Options = Options()) {
    self.options = options
    var delegateOptions = TfLiteCoreMlDelegateOptions()
    delegateOptions.enabled_devices = options.enabledDevices.cEnabledDevices
    delegateOptions.max_delegated_partitions = options.maxDelegatedPartitions
    delegateOptions.min_nodes_per_partition = options.minNodesPerPartition
    cDelegate = TfLiteCoreMlDelegateCreate(&delegateOptions)
    if cDelegate == nil {
      return nil
    }
  }

  deinit {
    TfLiteCoreMlDelegateDelete(cDelegate)
  }
}


extension CoreMLDelegate {
  /// Options for configuring the `CoreMLDelegate`.
  // TODO(b/143931022): Add preferred device support.
  public struct Options: Equatable, Hashable {
    /// A type determines Core ML delegate initialization on devices without Neural Engine. The
    /// default is .devicesWithNeuralEngine, where the delegate will not be created for
    /// devices that does not have Neural Engine.
    public var enabledDevices: CoreMLDelegateEnabledDevices = .devicesWithNeuralEngine
    /// Maximum number of Core ML delegates created.  Each graph corresponds to one delegated node
    /// subset in the TFLite model. Set this to 0 to delegate all possible partitions.
    public var maxDelegatedPartitions: Int32 = 0;

    // Minimum number of nodes per partition delegated with
    // Core ML delegate. Defaults to 2.
    public var  minNodesPerPartition: Int32 = 2;

    /// Creates a new instance with the default values.
    public init() {}
  }
}

/// A type determines Core ML delegate initialization on devices without Neural Engine.
public enum CoreMLDelegateEnabledDevices: Equatable, Hashable {
  /// Creates the delegate only for devices with Neural Engine.
  case devicesWithNeuralEngine
  /// Creates the delegate even when Neural Engine is not available.
  case allDevices

  /// The C `TfLiteCoreMlDelegateEnabledDevices` for the current `CoreMLDelegateEnabledDevices`.
  var cEnabledDevices: TfLiteCoreMlDelegateEnabledDevices {
    switch self {
    case .devicesWithNeuralEngine:
      return TfLiteCoreMlDelegateDevicesWithNeuralEngine
    case .allDevices:
      return TfLiteCoreMlDelegateAllDevices
    }
  }
}
