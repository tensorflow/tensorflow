/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// note: 100.01 is not representable in FP16 (is in FP32), so use 101.0 instead.
TEST_F(OpenCLOperationTest, MeanStddevNormSeparateBatches) {
  // zero mean, zero variance
  auto status = MeanStddevNormSeparateBatchesTest(0.0f, 0.0f, 0.0f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // zero mean, small variance
  status = MeanStddevNormSeparateBatchesTest(0.0f, 0.01f, 2.63e-4f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // zero mean, large variance
  status =
      MeanStddevNormSeparateBatchesTest(0.0f, 100.0f, 2.63e-4f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // small mean, zero variance
  status = MeanStddevNormSeparateBatchesTest(0.01f, 0.0f, 0.0f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // small mean, small variance
  status =
      MeanStddevNormSeparateBatchesTest(0.01f, 0.01f, 3.57e-4f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // small mean, large variance
  status =
      MeanStddevNormSeparateBatchesTest(1.0f, 100.0f, 2.63e-4f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // large mean, zero variance
  status = MeanStddevNormSeparateBatchesTest(100.0f, 0.0f, 0.0f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // large mean, small variance
  status =
      MeanStddevNormSeparateBatchesTest(100.0f, 1.0f, 2.63e-4f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // large mean, large variance
  status =
      MeanStddevNormSeparateBatchesTest(100.0f, 100.0f, 2.63e-4f, &exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, MeanStddevNormalizationAllBatches) {
  auto status = MeanStddevNormalizationAllBatchesTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, MeanStddevNormalizationLargeVector) {
  auto status = MeanStddevNormalizationLargeVectorTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
