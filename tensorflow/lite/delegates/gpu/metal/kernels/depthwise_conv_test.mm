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

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3_stride_h2_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface DepthwiseConvMetalTest : XCTestCase
@end

@implementation DepthwiseConvMetalTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)testDepthwiseConvSimpleWeights {
  auto status = DepthwiseConvSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConvNoMultiplier {
  auto status = DepthwiseConvNoMultiplierTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConvMultiplier2 {
  auto status = DepthwiseConvMultiplier2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConv3x3SimpleWeights {
  auto status = DepthwiseConv3x3SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConv3x3 {
  auto status = DepthwiseConv3x3Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthWiseConv3x3StrideH2SimpleWeights {
  auto status = DepthWiseConv3x3StrideH2SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
