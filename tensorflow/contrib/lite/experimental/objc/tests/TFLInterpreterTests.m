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

#import <XCTest/XCTest.h>

#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLInterpreterOptions.h"
#import "third_party/tensorflow/contrib/lite/experimental/objc/apis/TFLTensor.h"

NS_ASSUME_NONNULL_BEGIN

/** Model resource name. */
static NSString *const kAddModelResourceName = @"add";

/** Model resource type. */
static NSString *const kAddModelResourceType = @"bin";

/** Rank of the input and output tensor in the Add model. */
static const NSUInteger kAddModelTensorRank = 1U;

/** Size of the first (and only) dimension of the input and output tensor in the Add model. */
static const NSUInteger kAddModelTensorFirstDimensionSize = 2U;

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

/** Absolute path of the Add model resource. */
@property(nonatomic, nullable) NSString *modelPath;

/** Default interpreter using the Add model. */
@property(nonatomic, nullable) TFLInterpreter *interpreter;

@end

@implementation TFLInterpreterTests

#pragma mark - XCTestCase

- (void)setUp {
  [super setUp];

  NSBundle *bundle = [NSBundle bundleForClass:[self class]];
  self.modelPath = [bundle pathForResource:kAddModelResourceName ofType:kAddModelResourceType];
  self.interpreter = [[TFLInterpreter alloc] initWithModelPath:self.modelPath];
  XCTAssertNotNil(self.interpreter);
  XCTAssertTrue([self.interpreter allocateTensorsWithError:nil]);
}

- (void)tearDown {
  self.modelPath = nil;
  self.interpreter = nil;

  [super tearDown];
}

#pragma mark - Tests

- (void)testSuccessfulFullRun {
  // Shape for both input and output tensor.
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:kAddModelTensorFirstDimensionSize];

  // Creates the interpreter options.
  TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
  XCTAssertNotNil(options);
  options.numberOfThreads = 2;

  // Creates the interpreter.
  TFLInterpreter *customInterpreter = [[TFLInterpreter alloc] initWithModelPath:self.modelPath
                                                                        options:options];
  XCTAssertNotNil(customInterpreter);

  // Allocates memory for tensors.
  NSError *error;
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
  XCTAssertTrue([shape isEqualToArray:inputTensor.shape]);
  XCTAssertEqual(inputTensor.byteSize, sizeof(float) * kAddModelTensorFirstDimensionSize);

  // Copies the input data.
  NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
  float one = 1.f;
  float three = 3.f;
  [inputData appendBytes:&one length:sizeof(float)];
  [inputData appendBytes:&three length:sizeof(float)];
  XCTAssertTrue([customInterpreter copyData:inputData toInputTensorAtIndex:0 error:&error]);
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
  XCTAssertTrue([shape isEqualToArray:outputTensor.shape]);
  XCTAssertEqual(outputTensor.byteSize, sizeof(float) * kAddModelTensorFirstDimensionSize);

  // Tries to query an invalid output tensor index.
  TFLTensor *invalidOutputTensor = [customInterpreter outputTensorAtIndex:kInvalidOutputTensorIndex
                                                                    error:&error];
  XCTAssertNil(invalidOutputTensor);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidTensorIndex);

  // Gets the output tensor data.
  error = nil;
  NSData *outputData = [customInterpreter dataFromOutputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(outputData);
  XCTAssertNil(error);
  float output[kAddModelTensorFirstDimensionSize];
  [outputData getBytes:output length:(sizeof(float) * kAddModelTensorFirstDimensionSize)];
  XCTAssertEqualWithAccuracy(output[0], 3.f, kTestAccuracy);
  XCTAssertEqualWithAccuracy(output[1], 9.f, kTestAccuracy);
}

- (void)testInitWithModelPath_invalidPath {
  // Shape for both input and output tensor.
  NSMutableArray *shape = [NSMutableArray arrayWithCapacity:kAddModelTensorRank];
  shape[0] = [NSNumber numberWithUnsignedInteger:kAddModelTensorFirstDimensionSize];

  // Creates the interpreter.
  TFLInterpreter *brokenInterpreter = [[TFLInterpreter alloc] initWithModelPath:@"InvalidPath"];
  XCTAssertNotNil(brokenInterpreter);
  XCTAssertEqual(brokenInterpreter.inputTensorCount, 0);
  XCTAssertEqual(brokenInterpreter.outputTensorCount, 0);

  // Allocates memory for tensors.
  NSError *error;
  XCTAssertFalse([brokenInterpreter allocateTensorsWithError:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);

  // Resizes the intput tensor.
  XCTAssertFalse([brokenInterpreter resizeInputTensorAtIndex:0 toShape:shape error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);

  // Verifies the input tensor.
  TFLTensor *inputTensor = [brokenInterpreter inputTensorAtIndex:0 error:&error];
  XCTAssertNil(inputTensor);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);

  // Copies the input data.
  NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
  float one = 1.f;
  float three = 3.f;
  [inputData appendBytes:&one length:sizeof(float)];
  [inputData appendBytes:&three length:sizeof(float)];
  XCTAssertFalse([brokenInterpreter copyData:inputData toInputTensorAtIndex:0 error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);

  // Invokes the interpreter.
  XCTAssertFalse([brokenInterpreter invokeWithError:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);

  // Verifies the output tensor.
  TFLTensor *outputTensor = [brokenInterpreter outputTensorAtIndex:0 error:&error];
  XCTAssertNil(outputTensor);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);

  // Gets the output tensor data.
  NSData *outputData = [brokenInterpreter dataFromOutputTensorAtIndex:0 error:&error];
  XCTAssertNil(outputData);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeFailedToLoadModel);
}

- (void)testInvoke_beforeAllocation {
  TFLInterpreter *interpreterWithoutAllocation =
      [[TFLInterpreter alloc] initWithModelPath:self.modelPath];
  XCTAssertNotNil(interpreterWithoutAllocation);

  NSError *error;
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
  XCTAssertFalse([self.interpreter copyData:inputData toInputTensorAtIndex:0 error:&error]);
  XCTAssertEqual(error.code, TFLInterpreterErrorCodeInvalidInputByteSize);
}

@end

NS_ASSUME_NONNULL_END
