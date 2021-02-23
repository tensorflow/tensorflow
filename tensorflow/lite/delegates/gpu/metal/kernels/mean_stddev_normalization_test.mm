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
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface MeanStddevNormalizationTest : XCTestCase
@end

@implementation MeanStddevNormalizationTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

// note: 100.01 is not representable in FP16 (is in FP32), so use 101.0 instead.
- (void)testMeanStddevNormSeparateBatches {
  // zero mean, zero variance
  auto status = MeanStddevNormSeparateBatchesTest(0.0f, 0.0f, 0.0f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // zero mean, small variance
  status = MeanStddevNormSeparateBatchesTest(0.0f, 0.01f, 2.63e-4f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // zero mean, large variance
  status = MeanStddevNormSeparateBatchesTest(0.0f, 100.0f, 2.63e-4f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // small mean, zero variance
  status = MeanStddevNormSeparateBatchesTest(0.01f, 0.0f, 0.0f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // small mean, small variance
  status = MeanStddevNormSeparateBatchesTest(0.01f, 0.01f, 3.57e-4f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // small mean, large variance
  status = MeanStddevNormSeparateBatchesTest(1.0f, 100.0f, 2.63e-4f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // large mean, zero variance
  status = MeanStddevNormSeparateBatchesTest(100.0f, 0.0f, 0.0f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // large mean, small variance
  status = MeanStddevNormSeparateBatchesTest(100.0f, 1.0f, 2.63e-4f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  // large mean, large variance
  status = MeanStddevNormSeparateBatchesTest(100.0f, 100.0f, 2.63e-4f, &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMeanStddevNormalizationAllBatches {
  auto status = MeanStddevNormalizationAllBatchesTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMeanStddevNormalizationLargeVector {
  auto status = MeanStddevNormalizationLargeVectorTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
