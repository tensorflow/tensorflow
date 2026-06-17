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

#import "tensorflow/lite/objc/apis/TFLTensor.h"

#import "TFLErrorUtil.h"
#import "TFLInterpreter+Internal.h"
#import "TFLSignatureRunner+Internal.h"
#import "TFLTensor+Internal.h"

NS_ASSUME_NONNULL_BEGIN

// String names of input or output tensor types.
static NSString *const kTFLInputTensorTypeString = @"input";
static NSString *const kTFLOutputTensorTypeString = @"output";

@interface TFLTensor ()

// Redefines readonly properties.
@property(nonatomic) TFLTensorType type;
@property(nonatomic) NSUInteger index;
@property(nonatomic, nullable) NSString *nameInSignature;
@property(nonatomic, copy) NSString *name;
@property(nonatomic) TFLTensorDataType dataType;
@property(nonatomic, nullable) TFLQuantizationParameters *quantizationParameters;

/**
 * The tensor data accessor that could mutate the data on the tensor. It is either an interpreter or
 * a signature runner. It's a strong reference to ensure that the interpreter or the signature
 * runner is never released before this tensor is released.
 *
 * @warning Never let the interpreter or the signature runner hold a strong reference to the tensor
 * to avoid retain cycles.
 */
@property(nonatomic, strong) id<TFLTensorDataAccessor> tensorDataAccessor;

@end

@implementation TFLTensor

#pragma mark - Public

- (BOOL)copyData:(NSData *)data error:(NSError **)error {
  return [self.tensorDataAccessor copyData:data toInputTensor:self error:error];
}

- (nullable NSData *)dataWithError:(NSError **)error {
  return [self.tensorDataAccessor dataFromTensor:self error:error];
}

- (nullable NSArray<NSNumber *> *)shapeWithError:(NSError **)error {
  return [self.tensorDataAccessor shapeOfTensor:self error:error];
}

#pragma mark - TFLTensor (Internal)

- (instancetype)initWithInterpreter:(TFLInterpreter *)interpreter
                               type:(TFLTensorType)type
                              index:(NSUInteger)index
                               name:(NSString *)name
                           dataType:(TFLTensorDataType)dataType
             quantizationParameters:(nullable TFLQuantizationParameters *)quantizationParameters {
  self = [super init];
  if (self != nil) {
    _tensorDataAccessor = interpreter;
    _type = type;
    _index = index;
    _name = [name copy];
    _dataType = dataType;
    _quantizationParameters = quantizationParameters;
  }
  return self;
}

- (instancetype)initWithSignatureRunner:(TFLSignatureRunner *)signatureRunner
                                   type:(TFLTensorType)type
                        nameInSignature:(NSString *)nameInSignature
                                   name:(NSString *)name
                               dataType:(TFLTensorDataType)dataType
                 quantizationParameters:
                     (nullable TFLQuantizationParameters *)quantizationParameters {
  self = [super init];
  if (self != nil) {
    _tensorDataAccessor = signatureRunner;
    _type = type;
    _nameInSignature = [nameInSignature copy];
    _name = [name copy];
    _dataType = dataType;
    _quantizationParameters = quantizationParameters;
  }
  return self;
}

+ (NSString *)stringForTensorType:(TFLTensorType)type {
  switch (type) {
    case TFLTensorTypeInput:
      return kTFLInputTensorTypeString;
    case TFLTensorTypeOutput:
      return kTFLOutputTensorTypeString;
  }
}

@end

NS_ASSUME_NONNULL_END
