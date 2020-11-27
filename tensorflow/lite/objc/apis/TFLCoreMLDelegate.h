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

#import <Foundation/Foundation.h>

#import "TFLDelegate.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLCoreMLDelegateEnabledDevices
 * This enum specifies for which devices the Core ML delegate will be enabled.
 */
typedef NS_ENUM(NSUInteger, TFLCoreMLDelegateEnabledDevices) {
  /** Enables the delegate for devices with Neural Engine only. */
  TFLCoreMLDelegateEnabledDevicesNeuralEngine,
  /** Enables the delegate for all devices. */
  TFLCoreMLDelegateEnabledDevicesAll,
};

/** Custom configuration options for a Core ML delegate. */
@interface TFLCoreMLDelegateOptions : NSObject

/**
 * Indicates which devices the Core ML delegate should be enabled for. The default value is
 * `TFLCoreMLDelegateEnabledDevicesNeuralEngine`, indicating that the delegate is enabled for
 * Neural Engine devices only.
 */
@property(nonatomic) TFLCoreMLDelegateEnabledDevices enabledDevices;

/**
 * Target Core ML version for the model conversion. When it's not set, Core ML version will be set
 * to highest available version for the platform.
 */
@property(nonatomic) NSUInteger coreMLVersion;

/**
 * The maximum number of Core ML delegate partitions created. Each graph corresponds to one
 * delegated node subset in the TFLite model. The default value is `0` indicating that all possible
 * partitions are delegated.
 */
@property(nonatomic) NSUInteger maxDelegatedPartitions;

/**
 * The minimum number of nodes per partition to be delegated by the Core ML delegate. The default
 * value is `2`.
 */
@property(nonatomic) NSUInteger minNodesPerPartition;

@end

/** A delegate that uses the Core ML framework for performing TensorFlow Lite graph operations. */
@interface TFLCoreMLDelegate : TFLDelegate

/**
 * Initializes a new Core ML delegate with default options.
 *
 * @return A Core ML delegate initialized with default options. `nil` when the delegate creation
 * fails. For example, trying to initialize a Core ML delegate on an unsupported device.
 */
- (nullable instancetype)init;

/**
 * Initializes a new Core ML delegate with the given options.
 *
 * @param options Core ML delegate options.
 *
 * @return A Core ML delegate initialized with default options. `nil` when the delegate creation
 * fails. For example, trying to initialize Core ML delegate on an unsupported device.
 */
- (nullable instancetype)initWithOptions:(TFLCoreMLDelegateOptions *)options
    NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
