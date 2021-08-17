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

#ifdef COCOAPODS
@import TFLTensorFlowLite;
#else
#import "tensorflow/lite/objc/apis/TFLCoreMLDelegate.h"
#import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"
#endif

#import <XCTest/XCTest.h>

NS_ASSUME_NONNULL_BEGIN

/** Float model resource name.
 *  The model has a graph equivalent to (input + input) + input.
 */
static NSString* const kAddFloatModelResourceName = @"add";

/** Model resource type. */
static NSString* const kAddModelResourceType = @"bin";

/**
 * @var kTensorSize
 * Size of input and output tensors
 * @var kTensorChannels
 * Size of channel dimension of input and output tensors
 */

enum EnumType : int {kTensorSize = 8 * 8 * 3, kTensorChannels = 3};

/** Accuracy used in comparing floating numbers. */
static const float kTestAccuracy = 1E-5F;

@interface TFLCoreMLDelegateTests : XCTestCase
@end

@implementation TFLCoreMLDelegateTests

- (void)testCoreMLDelegate {
  if (@available(iOS 11.0, *)) {
  } else {
    return;
  }

  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  NSString* floatModelPath = [bundle pathForResource:kAddFloatModelResourceName
                                              ofType:kAddModelResourceType];

  TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
  TFLCoreMLDelegateOptions* coreMLOptions = [[TFLCoreMLDelegateOptions alloc] init];
  coreMLOptions.enabledDevices = TFLCoreMLDelegateEnabledDevicesAll;
  TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc] initWithOptions:coreMLOptions];
  XCTAssertNotNil(coreMLDelegate);

  NSError* error;
  TFLInterpreter* interpreter = [[TFLInterpreter alloc] initWithModelPath:floatModelPath
                                                                  options:options
                                                                delegates:@[ coreMLDelegate ]
                                                                    error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(interpreter);
  XCTAssertTrue([interpreter allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Copies the input data.
  NSMutableData* inputData = [NSMutableData dataWithLength:sizeof(float) * kTensorSize];
  for (int i = 0; i < kTensorSize / kTensorChannels; ++i) {
    float* data = (float*)inputData.mutableBytes;
    for (int j = 0; j < kTensorChannels; ++j) {
      data[i * kTensorChannels + j] = j;
    }
  }

  TFLTensor* inputTensor = [interpreter inputTensorAtIndex:0 error:&error];
  XCTAssertNotNil(inputTensor);
  XCTAssertTrue([inputTensor copyData:inputData error:&error]);
  XCTAssertNil(error);

  // Invokes the interpreter.
  XCTAssertTrue([interpreter invokeWithError:&error]);
  XCTAssertNil(error);

  // Gets the output tensor data.
  TFLTensor* outputTensor = [interpreter outputTensorAtIndex:0 error:&error];
  NSData* outputData = [outputTensor dataWithError:&error];
  XCTAssertNotNil(outputData);
  XCTAssertNil(error);

  float output[kTensorSize];
  [outputData getBytes:output length:(sizeof(float) * kTensorSize)];
  for (int i = 0; i < kTensorSize / kTensorChannels; ++i) {
    for (int j = 0; j < kTensorChannels; ++j) {
      XCTAssertEqualWithAccuracy(j * 3, output[i * kTensorChannels + j], kTestAccuracy);
    }
  }
}

@end

NS_ASSUME_NONNULL_END
