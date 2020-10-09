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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** Custom configuration options for a TensorFlow Lite interpreter. */
@interface TFLInterpreterOptions : NSObject

/**
 * Maximum number of threads that the interpreter should run on. Defaults to 0 (unspecified, letting
 * TensorFlow Lite to optimize the threading decision).
 */
@property(nonatomic) NSUInteger numberOfThreads;

/**
 * Experimental: Enable an optimized set of floating point CPU kernels (provided by XNNPACK).
 *
 * Enabling this flag will enable use of a new, highly optimized set of CPU kernels provided via the
 * XNNPACK delegate. Currently, this is restricted to a subset of floating point operations.
 * Eventually, we plan to enable this by default, as it can provide significant performance benefits
 * for many classes of floating point models. See
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
 * for more details.
 *
 * Things to keep in mind when enabling this flag:
 *
 *     * Startup time and resize time may increase.
 *     * Baseline memory consumption may increase.
 *     * Compatibility with other delegates (e.g., GPU) has not been fully validated.
 *     * Quantized models will not see any benefit.
 *
 * WARNING: This is an experimental interface that is subject to change.
 */
@property(nonatomic) BOOL useXNNPACK;

/**
 * Initializes a new instance of `TFLInterpreterOptions`.
 *
 * @return A new instance of `TFLInterpreterOptions`.
 */
- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
