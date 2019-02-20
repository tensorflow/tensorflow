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

#import <XCTest/XCTest.h>

#import "tensorflow/lite/experimental/objc/apis/TFLInterpreterOptions.h"
#import "tensorflow/lite/experimental/objc/apis/TFLQuantizationParameters.h"
#import "tensorflow/lite/experimental/objc/apis/TFLTensor.h"

NS_ASSUME_NONNULL_BEGIN

/** Float model resource name. */
static NSString *const kAddFloatModelResourceName = @"add";

/** Quantized model resource name. */
static NSString *const kAddQuantizedModelResourceName = @"add_quantized";

/** Model resource type. */
static NSString *const kAddModelResourceType = @"bin";

/** Rank of the input and output tensor in the Add model. */
static const NSUInteger kAddModelTensorRank = 1U;

/** Size of the first (and only) dimension of the input and output tensor in the Add model. */
static const NSUInteger kAddModelTensorFirstDimensionSize = 2U;

/** Quantization scale of the quantized model. */
static const float kAddQuantizedModelScale = 0.003922F;

/** Quantization zero point of the quantized model. */
static const int32_t kAddQuantizedModelZeroPoint = 0;

/** Invalid input tensor index. */
static const NSUInteger kInvalidInputTensorIndex = 1U;

/** Invalid output tensor index. */
static const NSUInteger kInvalidOutputTensorIndex = 1U;

/** Accurary used in comparing floating numbers. */
static const float kTestAccuracy = 1E-5F;

/**
 * Unit tests for TFLInterpreter.
 */
@interface TFLInterpreterTests : XCTestCase

/** Absolute path of the Add float model resource. */
@property(nonatomic, nullable) NSString *floatModelPath;

/** Default interpreter using the Add model. */
@property(nonatomic, nullable) TFLInterpreter *interpreter;

@end

@implementation TFLInterpreterTests

#pragma mark - XCTestCase

- (void)setUp {
  [super setUp];

  NSBundle *bundle = [NSBundle bundleForClass:[self class]];
  self.floatModelPath = [bundle pathForResource:kAddFloatModelResourceName
                                         ofType:kAddModelResourceType];
  NSError *error;
  self.interpreter = [[TFLInterpreter alloc] initWithModelPath:self.floatModelPath error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(self.interpreter);
  XCTAssertTrue([self.interpreter allocateTensorsWithError:nil]);
}

- (void)tearDown {
  self.floatModelPath = nil;
  self.interpreter = nil;

  [super tearDown];
}

#pragma mark - Tests

- (void)testSuccessfulFullRunAddFloatModel {
  // Shape for both input and output tensor.
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:kAddModelTensorFirstDimensionSize];

  // Creates the interpreter options.
  TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
  XCTAssertNotNil(options);
  options.numberOfThreads = 2;

  // Creates the interpreter.
  NSError *error;
  TFLInterpreter *customInterpreter = [[TFLInterpreter alloc] initWithModelPath:self.floatModelPath
                                                                        options:options
                                                                          error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(customInterpreter);

  // Allocates memory for tensors.
  XCTAssertTrue([customInterpreter allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Verifies input and output tensor counts.
  XCTAssertEqual(customInterpreter.inputTensorCount, 1);
  XCTAssertEqual(customInterpreter.outputTensorCount, 1);

  // Resizes the intput tensor.
  XCTAssertTrue([customInterpreter resizeInputTensorAtIndex:0 toShape:shape error:&error]);
  XCTAssertNil(error);

  // Re-allocates memory for tensors.
  XCTAssertTrue([customInterpreter allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Verifies the input tensor.
  TFLTensor *inputTensor = [customInterpreter inputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(inputTensor);
  XCTAssertNil(error);
  XCTAssertTrue([inputTensor.name isEqualToString:@"input"]);
  XCTAssertEqual(inputTensor.dataType, TFLTensorDataTypeFloat32);
  NSArray *inputTensorShape = [inputTensor shapeWithError:&error];
  XCTAssertNil(error);
  XCTAssertTrue([shape isEqualToArray:inputTensorShape]);

  // Copies the input data.
  NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
  float one = 1.f;
  float three = 3.f;
  [inputData appendBytes:&one length:sizeof(float)];
  [inputData appendBytes:&three length:sizeof(float)];
  XCTAssertTrue([inputTensor copyData:inputData error:&error]);
  XCTAssertNil(error);

  // Invokes the interpreter.
  XCTAssertTrue([customInterpreter invokeWithError:&error]);
  XCTAssertNil(error);

  // Verifies the output tensor.
  TFLTensor *outputTensor = [customInterpreter outputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(outputTensor);
  XCTAssertNil(error);
  XCTAssertTrue([outputTensor.name isEqualToString:@"output"]);
  XCTAssertEqual(outputTensor.dataType, TFLTensorDataTypeFloat32);
  NSArray *outputTensorShape = [outputTensor shapeWithError:&error];
  XCTAssertNil(error);
  XCTAssertTrue([shape isEqualToArray:outputTensorShape]);

  // Tries to query an invalid output tensor index.
  TFLTensor *invalidOutputTensor = [customInterpreter outputTensorAtIndex:kInvalidOutputTensorIndex
                                                                    error:&error];
  XCTAssertNil(invalidOutputTensor);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidTensorIndex);

  // Gets the output tensor data.
  error = nil;
  NSData *outputData = [outputTensor dataWithError:&error];
  XCTAssertNotNil(outputData);
  XCTAssertNil(error);
  float output[kAddModelTensorFirstDimensionSize];
  [outputData getBytes:output length:(sizeof(float) * kAddModelTensorFirstDimensionSize)];
  XCTAssertEqualWithAccuracy(output[0], 3.f, kTestAccuracy);
  XCTAssertEqualWithAccuracy(output[1], 9.f, kTestAccuracy);
}

- (void)testSuccessfulFullRunQuantizedModel {
  // Shape for both input and output tensor.
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:kAddModelTensorFirstDimensionSize];

  // Creates the interpreter options.
  TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
  XCTAssertNotNil(options);
  options.numberOfThreads = 2;

  NSBundle *bundle = [NSBundle bundleForClass:[self class]];
  NSString *quantizedModelPath = [bundle pathForResource:kAddQuantizedModelResourceName
                                                  ofType:kAddModelResourceType];

  // Creates the interpreter.
  NSError *error;
  TFLInterpreter *customInterpreter =
      [[TFLInterpreter alloc] initWithModelPath:quantizedModelPath options:options error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(customInterpreter);

  // Allocates memory for tensors.
  XCTAssertTrue([customInterpreter allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Verifies input and output tensor counts.
  XCTAssertEqual(customInterpreter.inputTensorCount, 1);
  XCTAssertEqual(customInterpreter.outputTensorCount, 1);

  // Resizes the intput tensor.
  XCTAssertTrue([customInterpreter resizeInputTensorAtIndex:0 toShape:shape error:&error]);
  XCTAssertNil(error);

  // Re-allocates memory for tensors.
  XCTAssertTrue([customInterpreter allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Verifies the input tensor.
  TFLTensor *inputTensor = [customInterpreter inputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(inputTensor);
  XCTAssertNil(error);
  XCTAssertTrue([inputTensor.name isEqualToString:@"input"]);
  XCTAssertEqual(inputTensor.dataType, TFLTensorDataTypeUInt8);
  XCTAssertEqualWithAccuracy(inputTensor.quantizationParameters.scale, kAddQuantizedModelScale,
                             kTestAccuracy);
  XCTAssertEqual(inputTensor.quantizationParameters.zeroPoint, kAddQuantizedModelZeroPoint);
  NSArray *inputTensorShape = [inputTensor shapeWithError:&error];
  XCTAssertNil(error);
  XCTAssertTrue([shape isEqualToArray:inputTensorShape]);

  // Copies the input data.
  NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
  uint8_t one = 1;
  uint8_t three = 3;
  [inputData appendBytes:&one length:sizeof(uint8_t)];
  [inputData appendBytes:&three length:sizeof(uint8_t)];
  XCTAssertTrue([inputTensor copyData:inputData error:&error]);
  XCTAssertNil(error);

  // Invokes the interpreter.
  XCTAssertTrue([customInterpreter invokeWithError:&error]);
  XCTAssertNil(error);

  // Verifies the output tensor.
  TFLTensor *outputTensor = [customInterpreter outputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(outputTensor);
  XCTAssertNil(error);
  XCTAssertTrue([outputTensor.name isEqualToString:@"output"]);
  XCTAssertEqual(outputTensor.dataType, TFLTensorDataTypeUInt8);
  XCTAssertEqualWithAccuracy(outputTensor.quantizationParameters.scale, kAddQuantizedModelScale,
                             kTestAccuracy);
  XCTAssertEqual(outputTensor.quantizationParameters.zeroPoint, kAddQuantizedModelZeroPoint);
  NSArray *outputTensorShape = [outputTensor shapeWithError:&error];
  XCTAssertNil(error);
  XCTAssertTrue([shape isEqualToArray:outputTensorShape]);

  // Tries to query an invalid output tensor index.
  TFLTensor *invalidOutputTensor = [customInterpreter outputTensorAtIndex:kInvalidOutputTensorIndex
                                                                    error:&error];
  XCTAssertNil(invalidOutputTensor);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidTensorIndex);

  // Gets the output tensor data.
  error = nil;
  NSData *outputData = [outputTensor dataWithError:&error];
  XCTAssertNotNil(outputData);
  XCTAssertNil(error);
  uint8_t output[kAddModelTensorFirstDimensionSize];
  [outputData getBytes:output length:(sizeof(uint8_t) * kAddModelTensorFirstDimensionSize)];
  XCTAssertEqual(output[0], 3);
  XCTAssertEqual(output[1], 9);
}

- (void)testInitWithModelPath_invalidPath {
  // Shape for both input and output tensor.
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:kAddModelTensorFirstDimensionSize];

  // Creates the interpreter.
  NSError *error;
  TFLInterpreter *brokenInterpreter = [[TFLInterpreter alloc] initWithModelPath:@"InvalidPath"
                                                                          error:&error];
  XCTAssertNil(brokenInterpreter);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);
}

- (void)testInvoke_beforeAllocation {
  NSError *error;
  TFLInterpreter *interpreterWithoutAllocation =
      [[TFLInterpreter alloc] initWithModelPath:self.floatModelPath error:&error];
  XCTAssertNotNil(interpreterWithoutAllocation);
  XCTAssertNil(error);

  XCTAssertFalse([interpreterWithoutAllocation invokeWithError:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToInvoke);
}

- (void)testInputTensorAtIndex_invalidIndex {
  NSError *error;
  TFLTensor *inputTensor = [self.interpreter inputTensorAtIndex:kInvalidInputTensorIndex
                                                          error:&error];
  XCTAssertNil(inputTensor);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidTensorIndex);
}

- (void)testResizeInputTensorAtIndex_invalidIndex {
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:kAddModelTensorFirstDimensionSize];
  NSError *error;
  XCTAssertFalse([self.interpreter resizeInputTensorAtIndex:kInvalidInputTensorIndex
                                                    toShape:shape
                                                      error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidTensorIndex);
}

- (void)testResizeInputTensorAtIndex_emptyShape {
  NSMutableArray *emptyShape = [NSMutableArray arrayWithCapacity:0];
  NSError *error;
  XCTAssertFalse([self.interpreter resizeInputTensorAtIndex:0 toShape:emptyShape error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidShape);
}

- (void)testResizeInputTensorAtIndex_zeroDimensionSize {
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:0];
  NSError *error;
  XCTAssertFalse([self.interpreter resizeInputTensorAtIndex:0 toShape:shape error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidShape);
}

- (void)testCopyDataToInputTensorAtIndex_invalidInputDataByteSize {
  NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
  float one = 1.f;
  float three = 3.f;
  [inputData appendBytes:&one length:sizeof(float)];
  [inputData appendBytes:&three length:(sizeof(float) - 1)];
  NSError *error;
  TFLTensor *inputTensor = [self.interpreter inputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(inputTensor);
  XCTAssertNil(error);
  XCTAssertFalse([inputTensor copyData:inputData error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidInputByteSize);
}

- (void)testCopyDataToOutputTensorAtIndex_notAllowed {
  NSMutableData *data = [NSMutableData dataWithCapacity:0];
  float one = 1.f;
  float three = 3.f;
  [data appendBytes:&one length:sizeof(float)];
  [data appendBytes:&three length:(sizeof(float) - 1)];
  NSError *error;
  TFLTensor *outputTensor = [self.interpreter outputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(outputTensor);
  XCTAssertNil(error);
  XCTAssertFalse([outputTensor copyData:data error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeCopyDataToOutputTensorNotAllowed);
}

@end

NS_ASSUME_NONNULL_END
