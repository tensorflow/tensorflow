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
#include "tensorflow/lite/delegates/gpu/common/tasks/padding_test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface PaddingTest : XCTestCase
@end

@implementation PaddingTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)testPaddingAppendWidth {
  auto status = PaddingAppendWidthTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingPrependWidth {
  auto status = PaddingPrependWidthTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingAppendHeight {
  auto status = PaddingAppendHeightTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingPrependHeight {
  auto status = PaddingPrependHeightTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingAppendChannels {
  auto status = PaddingAppendChannelsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingPrependChannels {
  auto status = PaddingPrependChannelsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingPrependChannelsX4 {
  auto status = PaddingPrependChannelsX4Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingComplex {
  auto status = PaddingComplexTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingReflectWidth {
  auto status = PaddingReflectWidthTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPaddingReflectChannels {
  auto status = PaddingReflectChannelsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
