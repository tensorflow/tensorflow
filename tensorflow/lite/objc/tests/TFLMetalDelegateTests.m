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
#import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"
#import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"
#endif

#import <Metal/MTLDevice.h>
#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Float model resource name.
 *  The model has four inputs (a, b, c, d) and two outputs (x, y)
 *  x = a + (b + c)
 *  y = (b + c) + d
 */
static NSString* const kAddFloatModelResourceName = @"multi_add";

/** Model resource type. */
static NSString* const kAddModelResourceType = @"bin";

/**
 * @var kTensorSize
 * Size of input and output tensors
 * @var kTensorChannels
 * Size of channel dimension of input and output tensors
 */

enum EnumType : int {kTensorSize = 8 * 8 * 3, kTensorChannels = 3};

/** Number of input tensors */
static const int kNumInputs = 4;

/** Number of output tensors */
static const int kNumOutputs = 2;

/** Accuracy used in comparing floating numbers. */
static const float kTestAccuracy = 1E-5F;

@interface TFLMetalDelegateTests : XCTestCase
@end

@implementation TFLMetalDelegateTests

- (void)testMetalDelegate {
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  NSString* floatModelPath = [bundle pathForResource:kAddFloatModelResourceName
                                              ofType:kAddModelResourceType];

  TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
  TFLMetalDelegate* metalDelegate = [[TFLMetalDelegate alloc] init];
  XCTAssertNotNil(metalDelegate);

  id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
  if (mtlDevice == nil) return;  // Stop testing if there's no GPU support

  NSError* error;
  TFLInterpreter* interpreter = [[TFLInterpreter alloc] initWithModelPath:floatModelPath
                                                                  options:options
                                                                delegates:@[ metalDelegate ]
                                                                    error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(interpreter);
  XCTAssertTrue([interpreter allocateTensorsWithError:&error]);
  XCTAssertNil(error);

  // Copies the input data. For each input, input[i, j, k] == k
  NSMutableData* inputData = [NSMutableData dataWithLength:sizeof(float) * kTensorSize];
  for (int i = 0; i < kTensorSize / kTensorChannels; ++i) {
    float* data = (float*)inputData.mutableBytes;
    for (int j = 0; j < kTensorChannels; ++j) {
      data[i * kTensorChannels + j] = j;
    }
  }

  for (int input_idx = 0; input_idx < kNumInputs; ++input_idx) {
    TFLTensor* inputTensor = [interpreter inputTensorAtIndex:input_idx error:&error];
    XCTAssertNotNil(inputTensor);
    XCTAssertTrue([inputTensor copyData:inputData error:&error]);
    XCTAssertNil(error);
  }

  // Invokes the interpreter.
  XCTAssertTrue([interpreter invokeWithError:&error]);
  XCTAssertNil(error);

  // Gets the output tensor data. For each output, output[i, j, k] == k * 3
  for (int output_idx = 0; output_idx < kNumOutputs; ++output_idx) {
    TFLTensor* outputTensor = [interpreter outputTensorAtIndex:output_idx error:&error];
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
}

@end

NS_ASSUME_NONNULL_END
