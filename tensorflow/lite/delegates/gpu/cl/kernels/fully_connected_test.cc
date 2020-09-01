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

#include "tensorflow/lite/delegates/gpu/cl/kernels/fully_connected.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, FullyConnected) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 4);
  attr.weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(creation_context_.GetDeviceInfo(), op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 1, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {14.5f, 37.5f}));
    }
  }
}

TEST_F(OpenCLOperationTest, RearrageWeights) {
  tflite::gpu::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(8, 1, 1, 8);
  weights.data = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  10.0, 11.0,
                  12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 20.0, 21.0, 22.0, 23.0,
                  24.0, 25.0, 26.0, 27.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
                  36.0, 37.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 60.0, 61.0,
                  62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 70.0, 71.0, 72.0, 73.0,
                  74.0, 75.0, 76.0, 77.0};

  std::vector<float> expected_rearranged_data = {
      0.0,  1.0,  2.0,  3.0,  10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0,
      23.0, 30.0, 31.0, 32.0, 33.0, 40.0, 41.0, 42.0, 43.0, 50.0, 51.0,
      52.0, 53.0, 60.0, 61.0, 62.0, 63.0, 70.0, 71.0, 72.0, 73.0, 4.0,
      5.0,  6.0,  7.0,  14.0, 15.0, 16.0, 17.0, 24.0, 25.0, 26.0, 27.0,
      34.0, 35.0, 36.0, 37.0, 44.0, 45.0, 46.0, 47.0, 54.0, 55.0, 56.0,
      57.0, 64.0, 65.0, 66.0, 67.0, 74.0, 75.0, 76.0, 77.0,
  };

  std::vector<float> data(8 * 8);
  float4* data_ptr = static_cast<float4*>(static_cast<void*>(data.data()));
  RearrangeFCWeightsToIOO4I4(weights, absl::MakeSpan(data_ptr, 8 * 8 / 4));

  EXPECT_THAT(data, ElementsAreArray(expected_rearranged_data));
}

TEST_F(OpenCLOperationTest, RearrageWeightsWhenPaddingIsRequired) {
  tflite::gpu::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(7, 1, 1, 7);
  weights.data = {
      0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
      26.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 40.0, 41.0,
      42.0, 43.0, 44.0, 45.0, 46.0, 50.0, 51.0, 52.0, 53.0, 54.0,
      55.0, 56.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0,
  };

  std::vector<float> expected_rearranged_data = {
      0.0,  1.0,  2.0,  3.0,  10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0,
      23.0, 30.0, 31.0, 32.0, 33.0, 40.0, 41.0, 42.0, 43.0, 50.0, 51.0,
      52.0, 53.0, 60.0, 61.0, 62.0, 63.0, 0.0,  0.0,  0.0,  0.0,  4.0,
      5.0,  6.0,  0.0,  14.0, 15.0, 16.0, 0.0,  24.0, 25.0, 26.0, 0.0,
      34.0, 35.0, 36.0, 0.0,  44.0, 45.0, 46.0, 0.0,  54.0, 55.0, 56.0,
      0.0,  64.0, 65.0, 66.0, 0.0,  0.0,  0.0,  0.0,  0.0,
  };

  std::vector<float> data(8 * 8);
  float4* data_ptr = static_cast<float4*>(static_cast<void*>(data.data()));
  RearrangeFCWeightsToIOO4I4(weights, absl::MakeSpan(data_ptr, 8 * 8 / 4));

  EXPECT_THAT(data, ElementsAreArray(expected_rearranged_data));
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
