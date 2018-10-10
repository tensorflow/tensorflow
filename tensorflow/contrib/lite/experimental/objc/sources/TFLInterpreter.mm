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

#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLInterpreter.h"

#import "TFLErrorUtil.h"
#import "TFLTensor+Internal.h"
#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLInterpreterOptions.h"
#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLTensor.h"

#include "third_party/tensorflow/contrib/lite/experimental/c/c_api.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLTensorType
 * This enum specifies input or output tensor types.
 */
typedef NS_ENUM(NSUInteger, TFLTensorType) {
  /** Input tensor type. */
  TFLTensorTypeInput,

  /** Output tensor type. */
  TFLTensorTypeOutput,
};

// Names used for indicating input or output in error messages.
static NSString *const kTFLInputDirection = @"input";
static NSString *const kTFLOutputDirection = @"output";

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

/**
 * An error in initializing the interpreter. If not `nil`, this error will be reported when the
 * interpreter is used.
 */
@property(nonatomic, nullable) NSError *initializationError;

@end

@implementation TFLInterpreter

#pragma mark - NSObject

- (void)dealloc {
  TFL_DeleteInterpreter(_interpreter);
}

#pragma mark - Public

- (instancetype)initWithModelPath:(NSString *)modelPath {
  return [self initWithModelPath:modelPath options:[[TFLInterpreterOptions alloc] init]];
}

- (instancetype)initWithModelPath:(NSString *)modelPath options:(TFLInterpreterOptions *)options {
  self = [super init];

  if (self != nil) {
    const char *modelPathCString = modelPath.UTF8String;
    NSString *pathErrorString =
        [NSString stringWithFormat:@"Cannot load model from path (%@).", modelPath];
    if (modelPathCString == nullptr) {
      _initializationError =
          [TFLErrorUtil interpreterErrorWithCode:TFLInterpreterErrorCodeFailedToLoadModel
                                     description:pathErrorString];
      return self;
    }

    TFL_Model *model = TFL_NewModelFromFile(modelPathCString);
    if (model == nullptr) {
      _initializationError =
          [TFLErrorUtil interpreterErrorWithCode:TFLInterpreterErrorCodeFailedToLoadModel
                                     description:pathErrorString];
      return self;
    }

    TFL_InterpreterOptions *cOptions = TFL_NewInterpreterOptions();
    if (cOptions == nullptr) {
      _initializationError =
          [TFLErrorUtil interpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                     description:@"Failed to create the interpreter."];
      TFL_DeleteModel(model);
      return self;
    }

    if (options.numberOfThreads > 0) {
      TFL_InterpreterOptionsSetNumThreads(cOptions, (int32_t)options.numberOfThreads);
    }
    TFL_InterpreterOptionsSetErrorReporter(cOptions, TFLInterpreterErrorReporter, nullptr);

    _interpreter = TFL_NewInterpreter(model, cOptions);
    if (_interpreter == nullptr) {
      _initializationError =
          [TFLErrorUtil interpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                     description:@"Failed to create the interpreter."];
    } else {
      _inputTensorCount = (NSUInteger)TFL_InterpreterGetInputTensorCount(_interpreter);
      _outputTensorCount = (NSUInteger)TFL_InterpreterGetOutputTensorCount(_interpreter);
      if (_inputTensorCount <= 0 || _outputTensorCount <= 0) {
        _initializationError =
            [TFLErrorUtil interpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCreateInterpreter
                                       description:@"Failed to create the interpreter."];
      }
    }
    TFL_DeleteInterpreterOptions(cOptions);
    TFL_DeleteModel(model);
  }

  return self;
}

- (BOOL)invokeWithError:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return NO;
  }

  if (TFL_InterpreterInvoke(self.interpreter) != kTfLiteOk) {
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToInvoke
                                   description:@"Failed to invoke the interpreter."
                                         error:error];
    return NO;
  }

  return YES;
}

- (nullable TFLTensor *)inputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return nil;
  }

  if (![self isValidTensorIndex:index belowLimit:self.inputTensorCount error:error]) {
    return nil;
  }

  return [self tensorOfType:TFLTensorTypeInput atIndex:index error:error];
}

- (nullable TFLTensor *)outputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return nil;
  }

  if (![self isValidTensorIndex:index belowLimit:self.outputTensorCount error:error]) {
    return nil;
  }

  return [self tensorOfType:TFLTensorTypeOutput atIndex:index error:error];
}

- (BOOL)resizeInputTensorAtIndex:(NSUInteger)index
                         toShape:(NSArray<NSNumber *> *)shape
                           error:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return NO;
  }

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
  for (int d = 0; d < shape.count; ++d) {
    int dimension = shape[d].intValue;
    if (dimension <= 0) {
      NSString *errorDescription = @"Invalid shape. Dimensions must be positive integers.";
      [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidShape
                                     description:errorDescription
                                           error:error];
      return NO;
    }
    cDimensions[d] = dimension;
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

- (BOOL)copyData:(NSData *)data toInputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return NO;
  }

  if (![self isValidTensorIndex:index belowLimit:self.inputTensorCount error:error]) {
    return NO;
  }

  TFL_Tensor *tensor = TFL_InterpreterGetInputTensor(self.interpreter, (int32_t)index);
  if (tensor == nullptr) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Failed to get input tensor at index (%lu).", (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToCopyDataToInputTensor
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  NSUInteger byteSize = (NSUInteger)TFL_TensorByteSize(tensor);
  if (data.length != byteSize) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Input tensor at index (%lu) expects data size (%lu), but got (%lu).",
                         (unsigned long)index, byteSize, (unsigned long)data.length];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeInvalidInputByteSize
                                   description:errorDescription
                                         error:error];
    return NO;
  }

  if (TFL_TensorCopyFromBuffer(tensor, data.bytes, data.length) != kTfLiteOk) {
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

- (nullable NSData *)dataFromOutputTensorAtIndex:(NSUInteger)index error:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return nil;
  }

  if (![self isValidTensorIndex:index belowLimit:self.outputTensorCount error:error]) {
    return nil;
  }

  const TFL_Tensor *tensor = TFL_InterpreterGetOutputTensor(self.interpreter, (int32_t)index);
  if (tensor == nullptr) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Failed to get output tensor at index (%lu).", (unsigned long)index];
    [TFLErrorUtil
        saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetDataFromOutputTensor
                         description:errorDescription
                               error:error];
    return nil;
  }

  void *bytes = TFL_TensorData(tensor);
  NSUInteger byteSize = (NSUInteger)TFL_TensorByteSize(tensor);
  if (bytes == nullptr || byteSize == 0) {
    NSString *errorDescription = [NSString
        stringWithFormat:@"Failed to get output tensor data at index (%lu).", (unsigned long)index];
    [TFLErrorUtil
        saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetDataFromOutputTensor
                         description:errorDescription
                               error:error];
    return nil;
  }

  return [NSData dataWithBytes:bytes length:byteSize];
}

- (BOOL)allocateTensorsWithError:(NSError **)error {
  if (self.initializationError != nil) {
    [self saveInitializationErrorToDestination:error];
    return NO;
  }

  if (TFL_InterpreterAllocateTensors(self.interpreter) != kTfLiteOk) {
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToAllocateTensors
                                   description:@"Failed to allocate memory for tensors."
                                         error:error];
    return NO;
  }
  return YES;
}

#pragma mark - Private

- (nullable TFLTensor *)tensorOfType:(TFLTensorType)type
                             atIndex:(NSUInteger)index
                               error:(NSError **)error {
  const TFL_Tensor *tensor = nullptr;
  NSString *tensorType;
  switch (type) {
    case TFLTensorTypeInput:
      tensor = TFL_InterpreterGetInputTensor(self.interpreter, (int32_t)index);
      tensorType = kTFLInputDirection;
      break;
    case TFLTensorTypeOutput:
      tensor = TFL_InterpreterGetOutputTensor(self.interpreter, (int32_t)index);
      tensorType = kTFLOutputDirection;
      break;
  }

  if (tensor == nullptr) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }

  const char *cName = TFL_TensorName(tensor);
  if (cName == nullptr) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get name of %@ tensor at index (%lu).", tensorType,
                                   (unsigned long)index];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }
  NSString *name = [NSString stringWithUTF8String:cName];

  TFLTensorDataType dataType = [self tensorDataTypeFromCTensorType:TFL_TensorType(tensor)];

  int32_t rank = TFL_TensorNumDims(tensor);
  if (rank <= 0) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"%@ tensor at index (%lu) has invalid rank (%d).", tensorType,
                                   (unsigned long)index, rank];
    [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetTensor
                                   description:errorDescription
                                         error:error];
    return nil;
  }
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:rank];
  for (int32_t d = 0; d < rank; d++) {
    int32_t dimension = TFL_TensorDim(tensor, d);
    if (dimension <= 0) {
      NSString *errorDescription =
          [NSString stringWithFormat:@"%@ tensor at index (%lu) has invalid %d-th dimension (%d).",
                                     tensorType, (unsigned long)index, d, dimension];
      [TFLErrorUtil saveInterpreterErrorWithCode:TFLInterpreterErrorCodeFailedToGetTensor
                                     description:errorDescription
                                           error:error];
      return nil;
    }
    shape[d] = @((NSUInteger)dimension);
  }

  // TODO: Set quantization parameters when C API supports it.
  return [[TFLTensor alloc] initWithName:name
                                dataType:dataType
                                   shape:shape
                                byteSize:(NSUInteger)TFL_TensorByteSize(tensor)
                  quantizationParameters:nil];
}

- (TFLTensorDataType)tensorDataTypeFromCTensorType:(TFL_Type)cTensorType {
  switch (cTensorType) {
    case kTfLiteFloat32:
      return TFLTensorDataTypeFloat32;
    case kTfLiteInt32:
      return TFLTensorDataTypeInt32;
    case kTfLiteUInt8:
      return TFLTensorDataTypeUInt8;
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

- (void)saveInitializationErrorToDestination:(NSError **)destination {
  if (destination != NULL) {
    *destination = self.initializationError;
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
