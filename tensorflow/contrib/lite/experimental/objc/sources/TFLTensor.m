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

#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLTensor.h"

#import "TFLTensor+Internal.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLTensor ()

// Redefines readonly properties.
@property(nonatomic, copy) NSString *name;
@property(nonatomic) TFLTensorDataType dataType;
@property(nonatomic, copy) NSArray<NSNumber *> *shape;
@property(nonatomic) NSUInteger byteSize;
@property(nonatomic, nullable) TFLQuantizationParameters *quantizationParameters;

@end

@implementation TFLTensor

#pragma mark - TFLTensor (Internal)

- (instancetype)initWithName:(NSString *)name
                    dataType:(TFLTensorDataType)dataType
                       shape:(NSArray<NSNumber *> *)shape
                    byteSize:(NSUInteger)byteSize
      quantizationParameters:(nullable TFLQuantizationParameters *)quantizationParameters {
  self = [super init];
  if (self != nil) {
    _name = [name copy];
    _dataType = dataType;
    _shape = [shape copy];
    _byteSize = byteSize;
    _quantizationParameters = quantizationParameters;
  }
  return self;
}

@end

NS_ASSUME_NONNULL_END
