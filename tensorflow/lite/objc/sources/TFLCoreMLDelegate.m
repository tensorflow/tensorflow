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

#import "tensorflow/lite/objc/apis/TFLCoreMLDelegate.h"

#ifdef COCOAPODS
@import TensorFlowLiteCCoreML;
#else
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#endif

NS_ASSUME_NONNULL_BEGIN

@implementation TFLCoreMLDelegateOptions

- (instancetype)init {
  self = [super init];
  if (self != nil) {
    _coreMLVersion = 0;
    _maxDelegatedPartitions = 0;
    _minNodesPerPartition = 2;
    _enabledDevices = TFLCoreMLDelegateEnabledDevicesNeuralEngine;
  }

  return self;
}

@end

@implementation TFLCoreMLDelegate

@synthesize cDelegate = _cDelegate;

#pragma mark - NSObject

- (void)dealloc {
  TfLiteCoreMlDelegateDelete((TfLiteDelegate*)self.cDelegate);
}

#pragma mark - Public

- (nullable instancetype)init {
  TFLCoreMLDelegateOptions* options = [[TFLCoreMLDelegateOptions alloc] init];
  return [self initWithOptions:options];
}

- (nullable instancetype)initWithOptions:(TFLCoreMLDelegateOptions*)options {
  self = [super init];
  if (self != nil) {
    TfLiteCoreMlDelegateOptions cOptions;

    cOptions.coreml_version = options.coreMLVersion;
    cOptions.max_delegated_partitions = options.maxDelegatedPartitions;
    cOptions.min_nodes_per_partition = options.minNodesPerPartition;

    switch (options.enabledDevices) {
      case TFLCoreMLDelegateEnabledDevicesNeuralEngine:
        cOptions.enabled_devices = TfLiteCoreMlDelegateDevicesWithNeuralEngine;
        break;

      case TFLCoreMLDelegateEnabledDevicesAll:
        cOptions.enabled_devices = TfLiteCoreMlDelegateAllDevices;
        break;
    }
    _cDelegate = TfLiteCoreMlDelegateCreate(&cOptions);
    if (_cDelegate == nil) {
      return nil;
    }
  }
  return self;
}

@end

NS_ASSUME_NONNULL_END
