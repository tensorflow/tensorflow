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

#import "tensorflow/lite/objc/apis/TFLInterpreter.h"
#import "tensorflow/lite/objc/apis/TFLQuantizationParameters.h"
#import "tensorflow/lite/objc/apis/TFLSignatureRunner.h"
#import "tensorflow/lite/objc/apis/TFLTensor.h"

#import <XCTest/XCTest.h>

/** Multiple signatures model resource name. */
static NSString *const kMultiSignaturesModelResourceName = @"multi_signatures";
/** Model resource type. */
static NSString *const kModelResourceType = @"bin";

static NSString *const kAddSignatureKey = @"add";
static NSString *const kSubSignatureKey = @"sub";
static NSString *const kDummySignatureKey = @"dummy";

@interface TFLSignatureRunnerTest : XCTestCase

/** Absolute path of the multi-signature model resource. */
@property(nonatomic, nullable) NSString *multiSignaturesModelPath;

@end

@implementation TFLSignatureRunnerTest

- (void)setUp {
  [super setUp];

  NSBundle *bundle = [NSBundle bundleForClass:[self class]];
  self.multiSignaturesModelPath = [bundle pathForResource:kMultiSignaturesModelResourceName
                                                   ofType:kModelResourceType];
}

- (void)tearDown {
  self.multiSignaturesModelPath = nil;
  [super tearDown];
}

- (void)testSignatureKeys {
  NSError *error;
  TFLInterpreter *interpreter =
      [[TFLInterpreter alloc] initWithModelPath:self.multiSignaturesModelPath error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(interpreter);
  NSArray<NSString *> *signatureKeys = interpreter.signatureKeys;
  NSArray<NSString *> *expectedKeys = @[ kAddSignatureKey, kSubSignatureKey ];
  XCTAssertTrue([signatureKeys isEqualToArray:expectedKeys]);

  // Validate signature runner for "add" signature.
  XCTAssertNotNil([interpreter signatureRunnerWithKey:kAddSignatureKey error:&error]);
  XCTAssertNil(error);

  // Test fail to get signature runner for dummy signature.
  XCTAssertNil([interpreter signatureRunnerWithKey:kDummySignatureKey error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, TFLSignatureRunnerErrorCodeFailedToCreateSignatureRunner);
}

- (void)testResizeInputTensor {
  NSError *error;
  TFLInterpreter *interpreter =
      [[TFLInterpreter alloc] initWithModelPath:self.multiSignaturesModelPath error:&error];
  TFLSignatureRunner *addRunner = [interpreter signatureRunnerWithKey:kAddSignatureKey
                                                                error:&error];
  XCTAssertNil(error);
  NSArray<NSString *> *expectedInputs = @[ @"x" ];
  XCTAssertTrue([addRunner.inputs isEqualToArray:expectedInputs]);

  // Validate signature "add" input tensor "x" before resizing.
  TFLTensor *inputTensor = [addRunner inputTensorWithName:@"x" error:&error];
  XCTAssertNotNil(inputTensor);
  XCTAssertEqual(inputTensor.dataType, TFLTensorDataTypeFloat32);
  XCTAssertTrue([[inputTensor shapeWithError:&error] isEqualToArray:@[ @(1) ]]);
  XCTAssertNil(error);
  XCTAssertEqual([inputTensor dataWithError:&error].length, 4U);
  XCTAssertNil(error);
  XCTAssertEqual(inputTensor.quantizationParameters.scale, 0.);
  XCTAssertEqual(inputTensor.quantizationParameters.zeroPoint, 0U);

  // Test fail to copy data before resizing the tensor
  float inputs[2] = {2.f, 4.f};
  NSData *inputData = [NSData dataWithBytes:&inputs length:(2 * sizeof(float))];
  XCTAssertFalse([inputTensor copyData:inputData error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, TFLSignatureRunnerErrorCodeInvalidInputByteSize);
  error = nil;

  // Resize signature "add" input tensor "x"
  NSArray<NSNumber *> *newShape = @[ @(2) ];
  XCTAssertTrue([addRunner resizeInputTensorWithName:@"x" toShape:newShape error:&error]);
  XCTAssertNil(error);
  XCTAssertTrue([addRunner allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Validate signature "add" input tensor "x" after resizing.
  inputTensor = [addRunner inputTensorWithName:@"x" error:&error];
  XCTAssertNotNil(inputTensor);
  XCTAssertEqual(inputTensor.dataType, TFLTensorDataTypeFloat32);
  XCTAssertTrue([[inputTensor shapeWithError:&error] isEqualToArray:newShape]);
  XCTAssertNil(error);
  XCTAssertEqual([inputTensor dataWithError:&error].length, 8U);
  XCTAssertNil(error);
  XCTAssertEqual(inputTensor.quantizationParameters.scale, 0.);
  XCTAssertEqual(inputTensor.quantizationParameters.zeroPoint, 0U);

  // Validate input tensor "x" after copying data
  XCTAssertTrue([inputTensor copyData:inputData error:&error]);
  XCTAssertNil(error);
  NSData *retrievedInputData = [inputTensor dataWithError:&error];
  XCTAssertNil(error);
  float retrievedInputs[2];
  [retrievedInputData getBytes:&retrievedInputs length:retrievedInputData.length];
  XCTAssertEqual(retrievedInputs[0], inputs[0]);
  XCTAssertEqual(retrievedInputs[1], inputs[1]);
}

- (void)testResizeInputTensor_invalidTensor {
  NSError *error;
  TFLInterpreter *interpreter =
      [[TFLInterpreter alloc] initWithModelPath:self.multiSignaturesModelPath error:&error];
  TFLSignatureRunner *addRunner = [interpreter signatureRunnerWithKey:kAddSignatureKey
                                                                error:&error];
  XCTAssertNil(error);

  // Test fail to get input tensor for a dummy input name.
  XCTAssertNil([addRunner inputTensorWithName:@"dummy" error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, TFLSignatureRunnerErrorCodeFailedToGetTensor);

  // Test fail to resize a dummy input tensor
  error = nil;
  XCTAssertFalse([addRunner resizeInputTensorWithName:@"dummy" toShape:@[ @(2) ] error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, TFLSignatureRunnerErrorCodeFailedToResizeInputTensor);
}

- (void)testInvokeWithInputs {
  NSError *error;
  TFLInterpreter *interpreter =
      [[TFLInterpreter alloc] initWithModelPath:self.multiSignaturesModelPath error:&error];
  TFLSignatureRunner *addRunner = [interpreter signatureRunnerWithKey:kAddSignatureKey
                                                                error:&error];
  XCTAssertNil(error);

  // Validate signature "add" output tensor "output_0" before inference
  NSArray<NSString *> *expectedOutputs = @[ @"output_0" ];
  XCTAssertTrue([addRunner.outputs isEqualToArray:expectedOutputs]);
  TFLTensor *outputTensor = [addRunner outputTensorWithName:@"output_0" error:&error];
  XCTAssertNotNil(outputTensor);
  XCTAssertEqual(outputTensor.dataType, TFLTensorDataTypeFloat32);
  XCTAssertTrue([[outputTensor shapeWithError:&error] isEqualToArray:@[ @(1) ]]);
  XCTAssertNil(error);
  XCTAssertEqual([outputTensor dataWithError:&error].length, 4U);
  XCTAssertNil(error);
  XCTAssertEqual(outputTensor.quantizationParameters.scale, 0.);
  XCTAssertEqual(outputTensor.quantizationParameters.zeroPoint, 0U);

  // Resize signature "add" input tensor "x"
  XCTAssertTrue([addRunner resizeInputTensorWithName:@"x" toShape:@[ @(2) ] error:&error]);
  XCTAssertNil(error);

  // Invoke signature "add" with inputs.
  float inputs[2] = {2.f, 4.f};
  NSData *inputData = [NSData dataWithBytes:&inputs length:(2 * sizeof(float))];
  XCTAssertTrue([addRunner invokeWithInputs:@{@"x" : inputData} Error:&error]);
  XCTAssertNil(error);

  // Validate signature "add" output tensor "output_0" after inference
  outputTensor = [addRunner outputTensorWithName:@"output_0" error:&error];
  XCTAssertNotNil(outputTensor);
  XCTAssertEqual(outputTensor.dataType, TFLTensorDataTypeFloat32);
  XCTAssertTrue([[outputTensor shapeWithError:&error] isEqualToArray:@[ @(2) ]]);
  XCTAssertNil(error);
  XCTAssertEqual(outputTensor.quantizationParameters.scale, 0.);
  XCTAssertEqual(outputTensor.quantizationParameters.zeroPoint, 0U);

  NSData *outputData = [outputTensor dataWithError:&error];
  XCTAssertNil(error);
  XCTAssertEqual(outputData.length, 8U);
  float outputs[2];
  [outputData getBytes:&outputs length:outputData.length];
  XCTAssertEqual(outputs[0], inputs[0] + 2.f);
  XCTAssertEqual(outputs[1], inputs[1] + 2.f);
}

- (void)testInvokeWithInputs_invalidInputs {
  NSError *error;
  TFLInterpreter *interpreter =
      [[TFLInterpreter alloc] initWithModelPath:self.multiSignaturesModelPath error:&error];
  TFLSignatureRunner *addRunner = [interpreter signatureRunnerWithKey:kAddSignatureKey
                                                                error:&error];
  XCTAssertNil(error);

  // Invoke signature "add" with invalid input data.
  float inputs[2] = {2.f, 4.f};
  NSData *inputData = [NSData dataWithBytes:&inputs length:(2 * sizeof(float))];
  XCTAssertFalse([addRunner invokeWithInputs:@{@"x" : inputData} Error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, TFLSignatureRunnerErrorCodeInvalidInputByteSize);

  // Invoke signature "add" with invalid input name.
  error = nil;
  float input = 2.f;
  inputData = [NSData dataWithBytes:&input length:(1 * sizeof(float))];
  XCTAssertFalse([addRunner invokeWithInputs:@{@"dummy" : inputData} Error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, TFLSignatureRunnerErrorCodeFailedToGetTensor);
}

@end
