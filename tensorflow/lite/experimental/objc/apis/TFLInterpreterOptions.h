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
 * Initializes a new instance of `TFLInterpreterOptions`.
 *
 * @return A new instance of `TFLInterpreterOptions`.
 */
- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
