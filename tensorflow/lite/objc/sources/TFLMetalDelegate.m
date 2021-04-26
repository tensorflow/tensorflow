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

#import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"

#ifdef COCOAPODS
@import TensorFlowLiteCMetal;
#else
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif

NS_ASSUME_NONNULL_BEGIN

@implementation TFLMetalDelegateOptions

#pragma mark - Public

- (instancetype)init {
  self = [super init];
  if (self != nil) {
    _quantizationEnabled = true;
    _waitType = TFLMetalDelegateThreadWaitTypePassive;
  }
  return self;
}

@end

@implementation TFLMetalDelegate

@synthesize cDelegate = _cDelegate;

#pragma mark - NSObject

- (void)dealloc {
  TFLGpuDelegateDelete(self.cDelegate);
}

#pragma mark - Public

- (nullable instancetype)init {
  TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
  return [self initWithOptions:options];
}

- (nullable instancetype)initWithOptions:(TFLMetalDelegateOptions*)options {
  self = [super init];
  if (self != nil) {
    TFLGpuDelegateOptions cOptions;
    cOptions.allow_precision_loss = options.precisionLossAllowed;
    cOptions.enable_quantization = options.quantizationEnabled;
    switch (options.waitType) {
      case TFLMetalDelegateThreadWaitTypeDoNotWait:
        cOptions.wait_type = TFLGpuDelegateWaitTypeDoNotWait;
        break;
      case TFLMetalDelegateThreadWaitTypePassive:
        cOptions.wait_type = TFLGpuDelegateWaitTypePassive;
        break;
      case TFLMetalDelegateThreadWaitTypeActive:
        cOptions.wait_type = TFLGpuDelegateWaitTypeActive;
        break;
      case TFLMetalDelegateThreadWaitTypeAggressive:
        cOptions.wait_type = TFLGpuDelegateWaitTypeAggressive;
        break;
    }
    _cDelegate = TFLGpuDelegateCreate(&cOptions);
    if (_cDelegate == nil) {
      return nil;
    }
  }
  return self;
}

@end

NS_ASSUME_NONNULL_END
