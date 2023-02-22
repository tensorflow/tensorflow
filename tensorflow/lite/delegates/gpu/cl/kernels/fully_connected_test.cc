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

#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

using ::testing::ElementsAreArray;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, FullyConnected) {
  auto status = FullyConnectedTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, FullyConnectedLarge) {
  auto status = FullyConnectedLargeTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, FullyConnectedExtraLarge) {
  auto status = FullyConnectedExtraLargeTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, FullyConnectedInt8) {
  auto status = FullyConnectedInt8Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, RearrageWeights) {
  tflite::gpu::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(8, 1, 1, 8);
  weights.data = {
      0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,   //
      10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,  //
      20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f,  //
      30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f,  //
      40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,  //
      50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f,  //
      60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f,  //
      70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f   //
  };

  std::vector<float> expected_rearranged_data = {
      // Top-left block
      0.0f, 10.0f, 20.0f, 30.0f, 1.0f, 11.0f, 21.0f, 31.0f, 2.0f, 12.0f, 22.0f,
      32.0f, 3.0f, 13.0f, 23.0f, 33.0f,
      // Bottom-left block
      40.0f, 50.0f, 60.0f, 70.0f, 41.0f, 51.0f, 61.0f, 71.0f, 42.0f, 52.0f,
      62.0f, 72.0f, 43.0f, 53.0f, 63.0f, 73.0f,
      // Top-right block
      4.0f, 14.0f, 24.0f, 34.0f, 5.0f, 15.0f, 25.0f, 35.0f, 6.0f, 16.0f, 26.0f,
      36.0f, 7.0f, 17.0f, 27.0f, 37.0f,
      // Bottom-right block
      44.0f, 54.0f, 64.0f, 74.0f, 45.0f, 55.0f, 65.0f, 75.0f, 46.0f, 56.0f,
      66.0f, 76.0f, 47.0f, 57.0f, 67.0f, 77.0f};

  std::vector<float> data(8 * 8);
  RearrangeFCWeightsToIOO4I4(weights, data.data());

  EXPECT_THAT(data, ElementsAreArray(expected_rearranged_data));
}

TEST_F(OpenCLOperationTest, RearrageWeightsWhenPaddingIsRequired) {
  tflite::gpu::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(9, 1, 1, 7);
  weights.data = {
      0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,   //
      10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,  //
      20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f,  //
      30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f,  //
      40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f,  //
      50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,  //
      60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f,  //
      70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f,  //
      80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f,  //
  };

  std::vector<float> expected_rearranged_data = {
      // Top-left block
      0.0f, 10.0f, 20.0f, 30.0f, 1.0f, 11.0f, 21.0f, 31.0f, 2.0f, 12.0f, 22.0f,
      32.0f, 3.0f, 13.0f, 23.0f, 33.0f,
      // Mid-left block
      40.0f, 50.0f, 60.0f, 70.0f, 41.0f, 51.0f, 61.0f, 71.0f, 42.0f, 52.0f,
      62.0f, 72.0f, 43.0f, 53.0f, 63.0f, 73.0f,
      // Bottom-left block
      80.0f, 0.0f, 0.0f, 0.0f, 81.0f, 0.0f, 0.0f, 0.0f, 82.0f, 0.0f, 0.0f, 0.0f,
      83.0f, 0.0f, 0.0f, 0.0f,
      // Top-right block
      4.0f, 14.0f, 24.0f, 34.0f, 5.0f, 15.0f, 25.0f, 35.0f, 6.0f, 16.0f, 26.0f,
      36.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      // Mid-left block
      44.0f, 54.0f, 64.0f, 74.0f, 45.0f, 55.0f, 65.0f, 75.0f, 46.0f, 56.0f,
      66.0f, 76.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      // Bottom-right block
      84.0f, 0.0f, 0.0f, 0.0f, 85.0f, 0.0f, 0.0f, 0.0f, 86.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f};

  std::vector<float> data(12 * 8);
  RearrangeFCWeightsToIOO4I4(weights, data.data());

  EXPECT_THAT(data, ElementsAreArray(expected_rearranged_data));
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
