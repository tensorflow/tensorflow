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

#import "tensorflow/lite/objc/apis/TFLQuantizationParameters.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLQuantizationParameters (Internal)

/**
 * Initializes a `TFLQuantizationParameters` instance with the given scale and zero point.
 *
 * @param scale Scale of asymmetric quantization.
 * @param zeroPoint Zero point of asymmetric quantization.
 *
 * @return A new instance of `TFLQuantizationParameters` with the given scale and zero point.
 */
- (instancetype)initWithScale:(float)scale zeroPoint:(int32_t)zeroPoint;

@end

NS_ASSUME_NONNULL_END
