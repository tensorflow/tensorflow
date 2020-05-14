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

  /// Creates a new instance configured with the given `options`. Returns `nil` if the underlying
  /// Core ML delegate could not be created because `Options.enabledDevices` was set to
  /// `neuralEngine` but the device does not have the Neural Engine.
  ///
  /// - Parameters:
  ///   - options: Configurations for the delegate. The default is a new instance of
  ///       `CoreMLDelegate.Options` with the default configuration values.
  public init?(options: Options = Options()) {
    self.options = options
    var delegateOptions = TfLiteCoreMlDelegateOptions()
    delegateOptions.enabled_devices = options.enabledDevices.cEnabledDevices
    delegateOptions.coreml_version = Int32(options.coreMLVersion)
    delegateOptions.max_delegated_partitions = Int32(options.maxDelegatedPartitions)
    delegateOptions.min_nodes_per_partition = Int32(options.minNodesPerPartition)
    guard let delegate = TfLiteCoreMlDelegateCreate(&delegateOptions) else { return nil }
    cDelegate = delegate
  }

  deinit {
    TfLiteCoreMlDelegateDelete(cDelegate)
  }
}

extension CoreMLDelegate {
  /// A type indicating which devices the Core ML delegate should be enabled for.
  public enum EnabledDevices: Equatable, Hashable {
    /// Enables the delegate for devices with Neural Engine only.
    case neuralEngine
    /// Enables the delegate for all devices.
    case all

    /// The C `TfLiteCoreMlDelegateEnabledDevices` for the current `EnabledDevices`.
    var cEnabledDevices: TfLiteCoreMlDelegateEnabledDevices {
      switch self {
      case .neuralEngine:
        return TfLiteCoreMlDelegateDevicesWithNeuralEngine
      case .all:
        return TfLiteCoreMlDelegateAllDevices
      }
    }
  }

  /// Options for configuring the `CoreMLDelegate`.
  // TODO(b/143931022): Add preferred device support.
  public struct Options: Equatable, Hashable {
    /// A type indicating which devices the Core ML delegate should be enabled for. The default
    /// value is `.neuralEngine` indicating that the delegate is enabled for Neural Engine devices
    /// only.
    public var enabledDevices: EnabledDevices = .neuralEngine
    /// Target Core ML version for the model conversion. When it's not set, Core ML version will
    /// be set to highest available version for the platform.
    public var coreMLVersion = 0
    /// The maximum number of Core ML delegate partitions created. Each graph corresponds to one
    /// delegated node subset in the TFLite model. The default value is `0` indicating that all
    /// possible partitions are delegated.
    public var maxDelegatedPartitions = 0
    /// The minimum number of nodes per partition to be delegated by the Core ML delegate. The
    /// default value is `2`.
    public var minNodesPerPartition = 2

    /// Creates a new instance with the default values.
    public init() {}
  }
}
