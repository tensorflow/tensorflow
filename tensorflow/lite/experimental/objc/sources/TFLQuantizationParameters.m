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

#import "tensorflow/lite/experimental/objc/apis/TFLQuantizationParameters.h"

#import "TFLQuantizationParameters+Internal.h"

NS_ASSUME_NONNULL_BEGIN

@implementation TFLQuantizationParameters

#pragma mark - TFLTensor (Internal)

- (instancetype)initWithScale:(float)scale zeroPoint:(int32_t)zeroPoint {
  self = [super init];
  if (self != nil) {
    _scale = scale;
    _zeroPoint = zeroPoint;
  }
  return self;
}

@end

NS_ASSUME_NONNULL_END
