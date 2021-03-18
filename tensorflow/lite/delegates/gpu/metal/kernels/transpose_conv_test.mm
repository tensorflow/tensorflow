/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#import <XCTest/XCTest.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_3x3_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_3x3_thin_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_4x4_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin_test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface TransposeConvTest : XCTestCase
@end

@implementation TransposeConvTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)testConvolutionTransposedSimpleWeights {
  auto status = ConvolutionTransposedSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed {
  auto status = ConvolutionTransposedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed4x4SimpleWeights {
  auto status = ConvolutionTransposed4x4SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposedThinSimpleWeights {
  auto status = ConvolutionTransposedThinSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposedThin {
  auto status = ConvolutionTransposedThinTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed3x3ThinSimpleWeights {
  auto status = ConvolutionTransposed3x3ThinSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed3x3Thin {
  auto status = ConvolutionTransposed3x3ThinTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed3x3 {
  auto status = ConvolutionTransposed3x3Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
