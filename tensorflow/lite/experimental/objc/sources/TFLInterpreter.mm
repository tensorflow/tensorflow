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

#import "tensorflow/lite/experimental/objc/apis/TFLInterpreter.h"

#import "TFLErrorUtil.h"
#import "TFLQuantizationParameters+Internal.h"
#import "TFLTensor+Internal.h"
#import "tensorflow/lite/experimental/objc/apis/TFLInterpreterOptions.h"
#import "tensorflow/lite/experimental/objc/apis/TFLTensor.h"

#include "tensorflow/lite/experimental/c/c_api.h"

NS_ASSUME_NONNULL_BEGIN

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

/** TFL_Interpreter backed by C API. */
@property(nonatomic, nullable) TFL_Interpreter *interpreter;

@end

@implementation TFLInterpreter

#pragma mark - NSObject

- (void)dealloc {
  TFL_DeleteInterpreter(_interpreter);
}

#pragma mark - Public

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  return [self initWithModelPath:modelPath
                         options:[[TFLInterpreterOptions alloc] init]
                           error:error];
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   options:(TFLInterpreterOptions *)options
                                     error:(NSError **)error {
  self = [super init];

  if (self != nil) {
    TFL_Model *model = nullptr;
    TFL_InterpreterOptions *cOptions = nullptr;

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

      model = TFL_NewModelFromFile(modelPathCString);
      if (model == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToLoadModel
                                       description:pathErrorString
                                             error:error];
        return nil;
      }

      cOptions = TFL_NewInterpreterOptions();
      if (cOptions == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."
                                             error:error];
        return nil;
      }

      if (options.numberOfThreads > 0) {
        TFL_InterpreterOptionsSetNumThreads(cOptions, (int32_t)options.numberOfThreads);
      }
      TFL_InterpreterOptionsSetErrorReporter(cOptions, TFLInterpreterErrorReporter, nullptr);

      _interpreter = TFL_NewInterpreter(model, cOptions);
      if (_interpreter == nullptr) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."
                                             error:error];
        return nil;
      }

      _inputTensorCount = (NSUInteger)TFL_InterpreterGetInputTensorCount(_interpreter);
      _outputTensorCount = (NSUInteger)TFL_InterpreterGetOutputTensorCount(_interpreter);
      if (_inputTensorCount <= 0 || _outputTensorCount <= 0) {
        [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."
                                             error:error];
        return nil;
      }
    } @finally {
      TFL_DeleteInterpreterOptions(cOptions);
      TFL_DeleteModel(model);
    }
  }

  return self;
}

- (BOOL)invokeWithError:(NSError **)error {
  if (TFL_InterpreterInvoke(self.interpreter) != kTfLiteOk) {
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

  int cDimensions[self.inputTensorCount];
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

  if (TFL_InterpreterResizeInputTensor(self.interpreter, (int32_t)index, cDimensions,
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
  if (TFL_InterpreterAllocateTensors(self.interpreter) != kTfLiteOk) {
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToAllocateTensors
                                   description:@"Failed to allocate memory for tensors."
                                         error:error];
    return NO;
  }
  return YES;
}

#pragma mark - TFLInterpreter (Internal)

- (BOOL)copyData:(NSData *)data toInputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  const TFL_Tensor *cTensor = [self cTensorOfType:TFLTensorTypeInput atIndex:index error:error];
  if (cTensor == nullptr) {
    return NO;
  }

  NSUInteger byteSize = (NSUInteger)TFL_TensorByteSize(cTensor);
  if (data.length != byteSize) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Input tensor at index (%lu) expects data size (%lu), but got (%lu).",
                         (unsigned long)index, byteSize, (unsigned long)data.length];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidInputByteSize
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  if (TFL_TensorCopyFromBuffer((TFL_Tensor *)cTensor, data.bytes, data.length) != kTfLiteOk) {
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
  const TFL_Tensor *cTensor = [self cTensorOfType:tensor.type atIndex:tensor.index error:error];
  if (cTensor == nullptr) {
    return nil;
  }

  void *bytes = TFL_TensorData(cTensor);
  NSUInteger byteSize = (NSUInteger)TFL_TensorByteSize(cTensor);
  if (bytes == nullptr || byteSize == 0) {
    NSString *tensorType = [TFLTensor stringForTensorType:tensor.type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get data from %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetDataFromTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }

  return [NSData dataWithBytes:bytes length:byteSize];
}

- (nullable NSArray<NSNumber *> *)shapeOfTensor:(TFLTensor *)tensor error:(NSError **)error {
  const TFL_Tensor *cTensor = [self cTensorOfType:tensor.type atIndex:tensor.index error:error];
  if (cTensor == nullptr) {
    return nil;
  }

  NSString *tensorType = [TFLTensor stringForTensorType:tensor.type];
  int32_t rank = TFL_TensorNumDims(cTensor);
  if (rank <= 0) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"%@ tensor at index (%lu) has invalid rank (%d).", tensorType,
                                   (unsigned long)index, rank];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }

  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:rank];
  for (int32_t dimIndex = 0; dimIndex < rank; dimIndex++) {
    int32_t dimension = TFL_TensorDim(cTensor, dimIndex);
    if (dimension <= 0) {
      NSString *errorDescription =
          [NSString stringWithFormat:@"%@ tensor at index (%lu) has invalid %d-th dimension (%d).",
                                     tensorType, (unsigned long)index, dimIndex, dimension];
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

- (const TFL_Tensor *)cTensorOfType:(TFLTensorType)type
                            atIndex:(NSUInteger)index
                              error:(NSError **)error {
  const TFL_Tensor *tensor = nullptr;

  switch (type) {
    case TFLTensorTypeInput:
      tensor = TFL_InterpreterGetInputTensor(self.interpreter, (int32_t)index);
      break;
    case TFLTensorTypeOutput:
      tensor = TFL_InterpreterGetOutputTensor(self.interpreter, (int32_t)index);
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
  const TFL_Tensor *tensor = [self cTensorOfType:type atIndex:index error:error];

  if (tensor == nullptr) {
    return nil;
  }

  NSString *tensorType = [TFLTensor stringForTensorType:type];
  const char *cName = TFL_TensorName(tensor);
  if (cName == nullptr) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get name of %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }
  NSString *name = [NSString stringWithUTF8String:cName];

  TFLTensorDataType dataType = [self tensorDataTypeFromCTensorType:TFL_TensorType(tensor)];

  TFL_QuantizationParams cParams = TFL_TensorQuantizationParams(tensor);
  TFLQuantizationParameters *quantizationParams;

  // TODO(b/119735362): Update this check once the TFL_QuantizationParams struct has a mode.
  if (cParams.scale != 0.0) {
    quantizationParams = [[TFLQuantizationParameters alloc] initWithScale:cParams.scale
                                                                zeroPoint:cParams.zero_point];
  }

  // TODO: Set quantization parameters when C API supports it.
  return [[TFLTensor alloc] initWithInterpreter:self
                                           type:type
                                          index:index
                                           name:name
                                       dataType:dataType
                         quantizationParameters:quantizationParams];
}

- (TFLTensorDataType)tensorDataTypeFromCTensorType:(TFL_Type)cTensorType {
  switch (cTensorType) {
    case kTfLiteFloat32:
      return TFLTensorDataTypeFloat32;
    case kTfLiteInt32:
      return TFLTensorDataTypeInt32;
    case kTfLiteUInt8:
      return TFLTensorDataTypeUInt8;
    case kTfLiteInt8:
      return TFLTensorDataTypeInt8;
    case kTfLiteInt64:
      return TFLTensorDataTypeInt64;
    case kTfLiteBool:
      return TFLTensorDataTypeBool;
    case kTfLiteInt16:
      return TFLTensorDataTypeInt16;
    case kTfLiteNoType:
    case kTfLiteString:
    case kTfLiteComplex64:
      // kTfLiteString and kTfLiteComplex64 are not supported in TensorFlow Lite Objc API.
      return TFLTensorDataTypeNoType;
  }
}

- (BOOL)isValidTensorIndex:(NSUInteger)index
                belowLimit:(NSUInteger)totalTensorCount
                     error:(NSError **)error {
  if (index >= totalTensorCount) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Invalid tensor index (%lu) exceeds max (%lu).",
                                   (unsigned long)index, (unsigned long)(totalTensorCount - 1)];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidTensorIndex
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  return YES;
}

@end

NS_ASSUME_NONNULL_END
