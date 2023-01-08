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

#import "tensorflow/lite/objc/apis/TFLInterpreter.h"

#include <vector>

#import "TFLCommonUtil.h"
#import "TFLErrorUtil.h"
#import "TFLQuantizationParameters+Internal.h"
#import "TFLSignatureRunner+Internal.h"
#import "TFLTensor+Internal.h"
#import "tensorflow/lite/objc/apis/TFLDelegate.h"
#import "tensorflow/lite/objc/apis/TFLInterpreterOptions.h"
#import "tensorflow/lite/objc/apis/TFLTensor.h"

#ifdef COCOAPODS
#import <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_experimental.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif  // COCOAPODS

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSString *const TFLVersion =
    TfLiteVersion() == nullptr ? @"" : [NSString stringWithUTF8String:TfLiteVersion()];

/**
 * Error reporter for TFLInterpreter.
 *
 * @param user_data User data. Not used.
 * @param format Error message which may contain argument formatting specifiers.
 * @param args Values of the arguments in the error message.
 */
static void TFLInterpreterErrorReporter(void *user_data, const char *format, va_list args) {
  NSLog(@"%@", [[NSString alloc] initWithFormat:@(format) arguments:args]);
}

@interface TFLInterpreter ()

/** TfLiteInterpreter backed by C API. */
@property(nonatomic) TfLiteInterpreter *interpreter;

/** TfLiteDelegate backed by C API. */
@property(nonatomic, nullable) TfLiteDelegate *xnnPackDelegate;

@end

@implementation TFLInterpreter

@synthesize signatureKeys = _signatureKeys;

#pragma mark - NSObject

- (void)dealloc {
  TfLiteInterpreterDelete(_interpreter);
  TfLiteXNNPackDelegateDelete(_xnnPackDelegate);
}

#pragma mark - Public

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  return [self initWithModelPath:modelPath
                         options:[[TFLInterpreterOptions alloc] init]
                       delegates:@[]
                           error:error];
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   options:(TFLInterpreterOptions *)options
                                     error:(NSError **)error {
  return [self initWithModelPath:modelPath options:options delegates:@[] error:error];
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   options:(TFLInterpreterOptions *)options
                                 delegates:(NSArray<TFLDelegate *> *)delegates
                                     error:(NSError **)error {
  self = [super init];

  if (self != nil) {
    TfLiteModel *model = nullptr;
    TfLiteInterpreterOptions *cOptions = nullptr;

    @try {
      const char *modelPathCString = modelPath.UTF8String;
      NSString *pathErrorString =
          [NSString stringWithFormat:@"Cannot load model from path (%@).", modelPath];
      if (modelPathCString == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToLoadModel
                                       description:pathErrorString
                                             error:error];
        return nil;
      }

      model = TfLiteModelCreateFromFile(modelPathCString);
      if (model == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToLoadModel
                                       description:pathErrorString
                                             error:error];
        return nil;
      }

      cOptions = TfLiteInterpreterOptionsCreate();
      if (cOptions == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."
                                             error:error];
        return nil;
      }

      if (options.numberOfThreads > 0) {
        TfLiteInterpreterOptionsSetNumThreads(cOptions, (int32_t)options.numberOfThreads);
      }
      TfLiteInterpreterOptionsSetErrorReporter(cOptions, TFLInterpreterErrorReporter, nullptr);

      if (options.useXNNPACK) {
        TfLiteXNNPackDelegateOptions xnnPackOptions = TfLiteXNNPackDelegateOptionsDefault();
        if (options.numberOfThreads > 0) {
          xnnPackOptions.num_threads = (int32_t)options.numberOfThreads;
        }

        _xnnPackDelegate = TfLiteXNNPackDelegateCreate(&xnnPackOptions);
        TfLiteInterpreterOptionsAddDelegate(cOptions, _xnnPackDelegate);
      }

      for (TFLDelegate *delegate in delegates) {
        if (delegate.cDelegate != nullptr) {
          TfLiteInterpreterOptionsAddDelegate(
              cOptions, reinterpret_cast<TfLiteDelegate *>(delegate.cDelegate));
        }
      }

      TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, cOptions);
      if (interpreter == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."
                                             error:error];
        return nil;
      }
      _interpreter = interpreter;

      _inputTensorCount = (NSUInteger)TfLiteInterpreterGetInputTensorCount(_interpreter);
      _outputTensorCount = (NSUInteger)TfLiteInterpreterGetOutputTensorCount(_interpreter);
      if (_inputTensorCount <= 0 || _outputTensorCount <= 0) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."
                                             error:error];
        return nil;
      }
    } @finally {
      TfLiteInterpreterOptionsDelete(cOptions);
      TfLiteModelDelete(model);
    }
  }

  return self;
}

- (NSArray<NSString *> *)signatureKeys {
  if (_signatureKeys) return _signatureKeys;
  NSUInteger signatureCount = TfLiteInterpreterGetSignatureCount(self.interpreter);
  NSMutableArray<NSString *> *mutableKeyArray =
      [[NSMutableArray alloc] initWithCapacity:signatureCount];
  for (NSUInteger i = 0; i < signatureCount; i++) {
    const char *signatureNameCString =
        TfLiteInterpreterGetSignatureKey(self.interpreter, (int32_t)i);
    NSString *signatureName = @"";
    if (signatureNameCString != nullptr) {
      signatureName = [NSString stringWithUTF8String:signatureNameCString] ?: @"";
    }
    [mutableKeyArray addObject:signatureName];
  }
  _signatureKeys = [mutableKeyArray copy];
  return _signatureKeys;
}

- (BOOL)invokeWithError:(NSError **)error {
  if (TfLiteInterpreterInvoke(self.interpreter) != kTfLiteOk) {
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToInvoke
                                   description:@"Failed to invoke the interpreter."
                                         error:error];
    return NO;
  }

  return YES;
}

- (nullable TFLTensor *)inputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  if (![self isValidTensorIndex:index belowLimit:self.inputTensorCount error:error]) {
    return nil;
  }

  return [self tensorOfType:TFLTensorTypeInput atIndex:index error:error];
}

- (nullable TFLTensor *)outputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  if (![self isValidTensorIndex:index belowLimit:self.outputTensorCount error:error]) {
    return nil;
  }

  return [self tensorOfType:TFLTensorTypeOutput atIndex:index error:error];
}

- (BOOL)resizeInputTensorAtIndex:(NSUInteger)index
                         toShape:(NSArray<NSNumber *> *)shape
                           error:(NSError **)error {
  if (![self isValidTensorIndex:index belowLimit:self.inputTensorCount error:error]) {
    return NO;
  }

  if (shape.count == 0) {
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidShape
                                   description:@"Invalid shape. Must not be empty."
                                         error:error];
    return NO;
  }

  std::vector<int> cDimensions(shape.count);
  for (int dimIndex = 0; dimIndex < shape.count; ++dimIndex) {
    int dimension = shape[dimIndex].intValue;
    if (dimension <= 0) {
      NSString *errorDescription = @"Invalid shape. Dimensions must be positive integers.";
      [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidShape
                                     description:errorDescription
                                           error:error];
      return NO;
    }
    cDimensions[dimIndex] = dimension;
  }

  if (TfLiteInterpreterResizeInputTensor(self.interpreter, (int32_t)index, cDimensions.data(),
                                         (int32_t)shape.count) != kTfLiteOk) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Failed to resize input tensor at index (%lu).", (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToResizeInputTensor
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  return YES;
}

- (BOOL)allocateTensorsWithError:(NSError **)error {
  if (TfLiteInterpreterAllocateTensors(self.interpreter) != kTfLiteOk) {
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToAllocateTensors
                                   description:@"Failed to allocate memory for tensors."
                                         error:error];
    return NO;
  }
  return YES;
}

- (nullable TFLSignatureRunner *)signatureRunnerWithKey:(NSString *)key error:(NSError **)error {
  if (![self.signatureKeys containsObject:key]) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Failed to create a signature runner. Signature with key (%@) not found.",
                         key];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToCreateSignatureRunner
               description:errorDescription];
    return nil;
  }
  return [[TFLSignatureRunner alloc] initWithInterpreter:self signatureKey:key error:error];
}

#pragma mark - TFLTensorDataAccessor

- (BOOL)copyData:(NSData *)data toInputTensor:(TFLTensor *)inputTensor error:(NSError **)error {
  if (inputTensor.type == TFLTensorTypeOutput) {
    [TFLErrorUtil
        saveInterpreterErrorWithCode:TFLInterpreterErrorCodeCopyDataToOutputTensorNotAllowed
                         description:@"Cannot copy data into an output tensor."
                               error:error];
    return NO;
  }
  const TfLiteTensor *cTensor = [self cTensorOfType:TFLTensorTypeInput
                                            atIndex:inputTensor.index
                                              error:error];
  if (cTensor == nullptr) {
    return NO;
  }

  NSUInteger byteSize = (NSUInteger)TfLiteTensorByteSize(cTensor);
  if (data.length != byteSize) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Input tensor at index (%lu) expects data size (%lu), but got (%lu).",
                         (unsigned long)index, (unsigned long)byteSize, (unsigned long)data.length];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidInputByteSize
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  if (TfLiteTensorCopyFromBuffer((TfLiteTensor *)cTensor, data.bytes, data.length) != kTfLiteOk) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to copy data into input tensor at index (%lu).",
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCopyDataToInputTensor
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  return YES;
}

- (nullable NSData *)dataFromTensor:(TFLTensor *)tensor error:(NSError **)error {
  const TfLiteTensor *cTensor = [self cTensorOfType:tensor.type atIndex:tensor.index error:error];
  if (cTensor == nullptr) {
    return nil;
  }

  void *bytes = TfLiteTensorData(cTensor);
  NSUInteger byteSize = (NSUInteger)TfLiteTensorByteSize(cTensor);
  if (bytes == nullptr || byteSize == 0) {
    NSString *tensorType = [TFLTensor stringForTensorType:tensor.type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get data from %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)tensor.index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetDataFromTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }

  return [NSData dataWithBytes:bytes length:byteSize];
}

- (nullable NSArray<NSNumber *> *)shapeOfTensor:(TFLTensor *)tensor error:(NSError **)error {
  const TfLiteTensor *cTensor = [self cTensorOfType:tensor.type atIndex:tensor.index error:error];
  if (cTensor == nullptr) {
    return nil;
  }

  NSString *tensorType = [TFLTensor stringForTensorType:tensor.type];
  int32_t rank = TfLiteTensorNumDims(cTensor);
  if (rank <= 0) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"%@ tensor at index (%lu) has invalid rank (%d).", tensorType,
                                   (unsigned long)tensor.index, rank];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }

  NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:rank];
  for (int32_t dimIndex = 0; dimIndex < rank; dimIndex++) {
    int32_t dimension = TfLiteTensorDim(cTensor, dimIndex);
    if (dimension <= 0) {
      NSString *errorDescription =
          [NSString stringWithFormat:@"%@ tensor at index (%lu) has invalid %d-th dimension (%d).",
                                     tensorType, (unsigned long)tensor.index, dimIndex, dimension];
      [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensor
                                     description:errorDescription
                                           error:error];
      return nil;
    }
    shape[dimIndex] = @((NSUInteger)dimension);
  }

  return shape;
}

#pragma mark - Private

- (const TfLiteTensor *)cTensorOfType:(TFLTensorType)type
                              atIndex:(NSUInteger)index
                                error:(NSError **)error {
  const TfLiteTensor *tensor = nullptr;

  switch (type) {
    case TFLTensorTypeInput:
      tensor = TfLiteInterpreterGetInputTensor(self.interpreter, (int32_t)index);
      break;
    case TFLTensorTypeOutput:
      tensor = TfLiteInterpreterGetOutputTensor(self.interpreter, (int32_t)index);
      break;
  }

  if (tensor == nullptr) {
    NSString *tensorType = [TFLTensor stringForTensorType:type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetTensor
                                   description:errorDescription
                                         error:error];
  }

  return tensor;
}

- (nullable TFLTensor *)tensorOfType:(TFLTensorType)type
                             atIndex:(NSUInteger)index
                               error:(NSError **)error {
  const TfLiteTensor *tensor = [self cTensorOfType:type atIndex:index error:error];

  if (tensor == nullptr) {
    return nil;
  }

  NSString *name = TFLTensorNameFromCTensor(tensor);
  if (!name) {
    NSString *tensorType = [TFLTensor stringForTensorType:type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get name of %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }
  TFLTensorDataType dataType = TFLTensorDataTypeFromCTensor(tensor);
  TFLQuantizationParameters *quantizationParams = TFLQuantizationParamsFromCTensor(tensor);

  return [[TFLTensor alloc] initWithInterpreter:self
                                           type:type
                                          index:index
                                           name:name
                                       dataType:dataType
                         quantizationParameters:quantizationParams];
}

- (BOOL)isValidTensorIndex:(NSUInteger)index
                belowLimit:(NSUInteger)totalTensorCount
                     error:(NSError **)error {
  if (index >= totalTensorCount) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Invalid tensor index (%lu) exceeds max (%lu).",
                                   (unsigned long)index, (totalTensorCount - 1)];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensorIndex
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  return YES;
}

@end

NS_ASSUME_NONNULL_END
