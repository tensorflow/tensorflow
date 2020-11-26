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
 * @enum TFLMetalDelegateThreadWaitType
 * This enum specifies wait type for Metal delegate.
 */
typedef NS_ENUM(NSUInteger, TFLMetalDelegateThreadWaitType) {

  /**
   * The thread does not wait for the work to complete. Useful when the output of the work is used
   * with the GPU pipeline.
   */
  TFLMetalDelegateThreadWaitTypeDoNotWait,
  /** The thread waits until the work is complete. */
  TFLMetalDelegateThreadWaitTypePassive,
  /**
   * The thread waits for the work to complete with minimal latency, which may require additional
   * CPU resources.
   */
  TFLMetalDelegateThreadWaitTypeActive,
  /** The thread waits for the work while trying to prevent the GPU from going into sleep mode. */
  TFLMetalDelegateThreadWaitTypeAggressive,
};

/** Custom configuration options for a Metal delegate. */
@interface TFLMetalDelegateOptions : NSObject

/**
 * Indicates whether the GPU delegate allows precision loss, such as allowing `Float16` precision
 * for a `Float32` computation. The default is `false`.
 */
@property(nonatomic, getter=isPrecisionLossAllowed) BOOL precisionLossAllowed;

/**
 * Indicates how the current thread should wait for work on the GPU to complete. The default
 * is `TFLMetalDelegateThreadWaitTypePassive`.
 */
@property(nonatomic) TFLMetalDelegateThreadWaitType waitType;

/**
 * Indicates whether the GPU delegate allows execution of an 8-bit quantized model. The default is
 * `true`.
 */
@property(nonatomic, getter=isQuantizationEnabled) BOOL quantizationEnabled;

@end

/**
 * A delegate that uses the `Metal` framework for performing TensorFlow Lite graph operations with
 * GPU acceleration.
 */
@interface TFLMetalDelegate : TFLDelegate

/**
 * Initializes a new GPU delegate with default options.
 *
 * @return A new GPU delegate with default options. `nil` when the GPU delegate creation fails.
 */
- (nullable instancetype)init;

/**
 * Initializes a new GPU delegate with the given options.
 *
 * @param options GPU delegate options.
 *
 * @return A new GPU delegate with default options. `nil` when the GPU delegate creation fails.
 */
- (nullable instancetype)initWithOptions:(TFLMetalDelegateOptions *)options
    NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
