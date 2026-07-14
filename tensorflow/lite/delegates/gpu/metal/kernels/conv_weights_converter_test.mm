/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface ConvWeightsConverterMetalTest : XCTestCase
@end

@implementation ConvWeightsConverterMetalTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)testConverterToConvWeights1x1OutX4 {
  const auto status = ConverterToConvWeights1x1OutX4Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConverterToConvWeights1x1OutX4Unaligned {
  const auto status = ConverterToConvWeights1x1OutX4UnalignedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConverterToConvWeights1x1OutX2 {
  const auto status = ConverterToConvWeights1x1OutX2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConverterToConvWeightsOutX2 {
  const auto status = ConverterToConvWeightsOutX2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConverterToConvTransposedWeights4x4 {
  const auto status = ConverterToConvTransposedWeights4x4Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConverterToConvWeights4xTextures {
  const auto status = ConverterToConvWeights4xTexturesTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
