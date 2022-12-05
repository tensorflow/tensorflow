// Copyright 2022 Google Inc. All rights reserved.
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

#import "TFLSignatureRunner+Internal.h"

#include <vector>

#import "TFLCommonUtil.h"
#import "TFLErrorUtil.h"
#import "TFLInterpreter+Internal.h"
#import "TFLQuantizationParameters+Internal.h"
#import "TFLTensor+Internal.h"

#include "tensorflow/lite/core/c/c_api_experimental.h"

NS_ASSUME_NONNULL_BEGIN

/** Domain for errors in the signature runner. */
NSErrorDomain const TFLSignatureRunnerErrorDomain = @"org.tensorflow.lite.SignatureRunner";

@interface TFLSignatureRunner ()

/**
 * The backing interpreter. It's a strong reference to ensure that the interpreter is never released
 * before this signature runner is released.
 *
 * @warning Never let the interpreter hold a strong reference to the signature runner to avoid
 * retain cycles.
 */
@property(nonatomic, readonly) TFLInterpreter *interpreter;

/** TfLiteSignatureRunner backed by C API. */
@property(nonatomic, readonly) TfLiteSignatureRunner *signatureRunner;

@end

@implementation TFLSignatureRunner {
  // Whether we need to allocate tensors memory.
  BOOL _isTensorsAllocationNeeded;
}

@synthesize inputs = _inputs;
@synthesize outputs = _outputs;
@synthesize signatureKey = _signatureKey;

#pragma mark - Initializer

- (nullable instancetype)initWithInterpreter:(TFLInterpreter *)interpreter
                                signatureKey:(NSString *)signatureKey
                                       error:(NSError **)error {
  self = [super init];
  if (self != nil) {
    _signatureKey = [signatureKey copy];
    const char *signatureKeyCString = _signatureKey.UTF8String;
    TfLiteSignatureRunner *signatureRunner =
        TfLiteInterpreterGetSignatureRunner(interpreter.interpreter, signatureKeyCString);
    if (signatureRunner == nullptr) {
      NSString *errorDescription =
          [NSString stringWithFormat:
                        @"Failed to create a signature runner. Signature with key (%@) not found.",
                        signatureKey];
      [TFLErrorUtil setError:error
                  withDomain:TFLSignatureRunnerErrorDomain
                        code:TFLSignatureRunnerErrorCodeFailedToCreateSignatureRunner
                 description:errorDescription];
      return nil;
    }
    _signatureRunner = signatureRunner;
    _interpreter = interpreter;
    _isTensorsAllocationNeeded = YES;
    [self allocateTensorsWithError:error];
  }
  return self;
}

- (void)dealloc {
  TfLiteSignatureRunnerDelete(_signatureRunner);
}

#pragma mark - Public

- (NSArray<NSString *> *)inputs {
  if (_inputs) return _inputs;
  NSUInteger inputCount = TfLiteSignatureRunnerGetInputCount(self.signatureRunner);
  NSMutableArray<NSString *> *mutableInputsArray =
      [[NSMutableArray alloc] initWithCapacity:inputCount];
  for (NSUInteger i = 0; i < inputCount; i++) {
    const char *inputNameCString =
        TfLiteSignatureRunnerGetInputName(self.signatureRunner, (int32_t)i);
    NSString *inputName = @"";
    if (inputNameCString != nullptr) {
      inputName = [NSString stringWithUTF8String:inputNameCString] ?: @"";
    };
    [mutableInputsArray addObject:inputName];
  }
  _inputs = [mutableInputsArray copy];
  return _inputs;
}

- (NSArray<NSString *> *)outputs {
  if (_outputs) return _outputs;
  NSUInteger outputCount = TfLiteSignatureRunnerGetOutputCount(self.signatureRunner);
  NSMutableArray<NSString *> *mutableOutputsArray =
      [[NSMutableArray alloc] initWithCapacity:outputCount];
  for (NSUInteger i = 0; i < outputCount; i++) {
    const char *outputNameCString =
        TfLiteSignatureRunnerGetOutputName(self.signatureRunner, (int32_t)i);
    NSString *outputName = @"";
    if (outputNameCString != nullptr) {
      outputName = [NSString stringWithUTF8String:outputNameCString] ?: @"";
    }
    [mutableOutputsArray addObject:outputName];
  }
  _outputs = [mutableOutputsArray copy];
  return _outputs;
}

- (nullable TFLTensor *)inputTensorWithName:(NSString *)name error:(NSError **)error {
  return [self tensorOfType:TFLTensorTypeInput nameInSignature:name error:error];
}

- (nullable TFLTensor *)outputTensorWithName:(NSString *)name error:(NSError **)error {
  return [self tensorOfType:TFLTensorTypeOutput nameInSignature:name error:error];
}

- (BOOL)resizeInputTensorWithName:(NSString *)name
                          toShape:(NSArray<NSNumber *> *)shape
                            error:(NSError **)error {
  if (shape.count == 0) {
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeInvalidShape
               description:@"Invalid shape. Must not be empty."];
    return NO;
  }

  std::vector<int> cDimensions(shape.count);
  for (int dimIndex = 0; dimIndex < shape.count; ++dimIndex) {
    int dimension = shape[dimIndex].intValue;
    if (dimension <= 0) {
      [TFLErrorUtil setError:error
                  withDomain:TFLSignatureRunnerErrorDomain
                        code:TFLSignatureRunnerErrorCodeInvalidShape
                 description:@"Invalid shape. Dimensions must be positive integers."];
      return NO;
    }
    cDimensions[dimIndex] = dimension;
  }

  if (TfLiteSignatureRunnerResizeInputTensor(self.signatureRunner, name.UTF8String,
                                             cDimensions.data(),
                                             (int32_t)shape.count) != kTfLiteOk) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to resize input tensor with input name (%@).", name];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToResizeInputTensor
               description:errorDescription];
    return NO;
  }

  // Need to reallocate tensor memory.
  _isTensorsAllocationNeeded = YES;

  return YES;
}

- (BOOL)allocateTensorsWithError:(NSError **)error {
  if (!_isTensorsAllocationNeeded) return YES;
  if (TfLiteSignatureRunnerAllocateTensors(self.signatureRunner) != kTfLiteOk) {
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToAllocateTensors
               description:@"Failed to allocate memory for tensors."];
    return NO;
  }
  _isTensorsAllocationNeeded = NO;
  return YES;
}

- (BOOL)invokeWithInputs:(NSDictionary<NSString *, NSData *> *)inputs Error:(NSError **)error {
  if (![self allocateTensorsWithError:error]) return NO;

  // Fill in input data.
  for (NSString *inputName in inputs.allKeys) {
    TFLTensor *inputTensor = [self inputTensorWithName:inputName error:error];
    if (!inputTensor) return NO;
    if (![inputTensor copyData:inputs[inputName] error:error]) return NO;
  }

  if (TfLiteSignatureRunnerInvoke(self.signatureRunner) != kTfLiteOk) {
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToInvoke
               description:@"Failed to invoke the signature runner."];
    return NO;
  }

  return YES;
}

#pragma mark - TFLTensorDataAccessor

- (BOOL)copyData:(NSData *)data toInputTensor:(TFLTensor *)inputTensor error:(NSError **)error {
  if (inputTensor.type == TFLTensorTypeOutput) {
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeCopyDataToOutputTensorNotAllowed
               description:@"Cannot copy data into an output tensor."];
    return NO;
  }
  const TfLiteTensor *cTensor = [self cTensorOfType:TFLTensorTypeInput
                                    nameInSignature:inputTensor.nameInSignature
                                              error:error];
  if (cTensor == nullptr) {
    return NO;
  }

  NSUInteger byteSize = (NSUInteger)TfLiteTensorByteSize(cTensor);
  if (data.length != byteSize) {
    NSString *errorDescription = [NSString
        stringWithFormat:
            @"Input tensor with input name (%@) expects data size (%lu), but got (%lu).",
            inputTensor.nameInSignature, (unsigned long)byteSize, (unsigned long)data.length];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeInvalidInputByteSize
               description:errorDescription];
    return NO;
  }

  if (TfLiteTensorCopyFromBuffer((TfLiteTensor *)cTensor, data.bytes, data.length) != kTfLiteOk) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to copy data into input tensor with input name (%@).",
                                   inputTensor.nameInSignature];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToCopyDataToInputTensor
               description:errorDescription];
    return NO;
  }

  return YES;
}

- (nullable NSData *)dataFromTensor:(TFLTensor *)tensor error:(NSError **)error {
  const TfLiteTensor *cTensor = [self cTensorOfType:tensor.type
                                    nameInSignature:tensor.nameInSignature
                                              error:error];
  if (cTensor == nullptr) {
    return nil;
  }

  void *bytes = TfLiteTensorData(cTensor);
  NSUInteger byteSize = (NSUInteger)TfLiteTensorByteSize(cTensor);
  if (bytes == nullptr || byteSize == 0) {
    NSString *tensorType = [TFLTensor stringForTensorType:tensor.type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get data from %@ tensor with %@ name (%@).",
                                   tensorType, tensorType, tensor.nameInSignature];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToGetDataFromTensor
               description:errorDescription];
    return nil;
  }

  return [NSData dataWithBytes:bytes length:byteSize];
}

- (nullable NSArray<NSNumber *> *)shapeOfTensor:(TFLTensor *)tensor error:(NSError **)error {
  const TfLiteTensor *cTensor = [self cTensorOfType:tensor.type
                                    nameInSignature:tensor.nameInSignature
                                              error:error];
  if (cTensor == nullptr) {
    return nil;
  }

  NSString *tensorType = [TFLTensor stringForTensorType:tensor.type];
  int32_t rank = TfLiteTensorNumDims(cTensor);
  if (rank <= 0) {
    NSString *errorDescription =
        [NSString stringWithFormat:@"%@ tensor with %@ name (%@) has invalid rank (%d).",
                                   tensorType, tensorType, tensor.nameInSignature, rank];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeInvalidTensor
               description:errorDescription];
    return nil;
  }

  NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:rank];
  for (int32_t dimIndex = 0; dimIndex < rank; dimIndex++) {
    int32_t dimension = TfLiteTensorDim(cTensor, dimIndex);
    if (dimension <= 0) {
      NSString *errorDescription = [NSString
          stringWithFormat:@"%@ tensor with %@ name (%@) has invalid %d-th dimension (%d).",
                           tensorType, tensorType, tensor.nameInSignature, dimIndex, dimension];
      [TFLErrorUtil setError:error
                  withDomain:TFLSignatureRunnerErrorDomain
                        code:TFLSignatureRunnerErrorCodeInvalidTensor
                 description:errorDescription];
      return nil;
    }
    shape[dimIndex] = @((NSUInteger)dimension);
  }

  return shape;
}

#pragma mark - Private

- (nullable TFLTensor *)tensorOfType:(TFLTensorType)type
                     nameInSignature:(NSString *)nameInSignature
                               error:(NSError **)error {
  const TfLiteTensor *tensor = [self cTensorOfType:type
                                   nameInSignature:nameInSignature
                                             error:error];

  if (tensor == nullptr) {
    return nil;
  }

  NSString *tensorName = TFLTensorNameFromCTensor(tensor);
  if (!tensorName) {
    NSString *tensorType = [TFLTensor stringForTensorType:type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get name of %@ tensor with %@ name (%@).",
                                   tensorType, tensorType, nameInSignature];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeInvalidTensor
               description:errorDescription];
    return nil;
  }
  TFLTensorDataType dataType = TFLTensorDataTypeFromCTensor(tensor);
  TFLQuantizationParameters *quantizationParams = TFLQuantizationParamsFromCTensor(tensor);

  return [[TFLTensor alloc] initWithSignatureRunner:self
                                               type:type
                                    nameInSignature:nameInSignature
                                               name:tensorName
                                           dataType:dataType
                             quantizationParameters:quantizationParams];
}

- (const TfLiteTensor *)cTensorOfType:(TFLTensorType)type
                      nameInSignature:(NSString *)nameInSignature
                                error:(NSError **)error {
  const TfLiteTensor *tensor = nullptr;
  const char *nameCString = nameInSignature.UTF8String;
  switch (type) {
    case TFLTensorTypeInput:
      tensor = TfLiteSignatureRunnerGetInputTensor(self.signatureRunner, nameCString);
      break;
    case TFLTensorTypeOutput:
      tensor = TfLiteSignatureRunnerGetOutputTensor(self.signatureRunner, nameCString);
      break;
  }

  if (tensor == nullptr) {
    NSString *tensorType = [TFLTensor stringForTensorType:type];
    NSString *errorDescription =
        [NSString stringWithFormat:@"Failed to get %@ tensor with %@ name (%@).", tensorType,
                                   tensorType, nameInSignature];
    [TFLErrorUtil setError:error
                withDomain:TFLSignatureRunnerErrorDomain
                      code:TFLSignatureRunnerErrorCodeFailedToGetTensor
               description:errorDescription];
  }

  return tensor;
}

@end

NS_ASSUME_NONNULL_END
