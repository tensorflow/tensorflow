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

#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Parameterized test: mean, difference, tolerance.
// Input is constructed as [mean-2*diff, mean-diff, mean+diff, mean+2*diff]
class MeanStddevNormalizationTest
    : public OpenCLOperationTest,
      public testing::WithParamInterface<std::tuple<float, float, float>> {};

TEST_P(MeanStddevNormalizationTest, SeparateBatches) {
  const float mean = std::get<0>(GetParam());
  const float diff = std::get<1>(GetParam());
  const float tolerance = std::get<2>(GetParam());

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {mean - 2 * diff, mean - diff, mean + diff,
                     mean + 2 * diff};
  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation =
          CreateMeanStdDevNormalization(op_def, env_.GetDevicePtr()->info_, 1);
      ASSERT_OK(ExecuteGPUOperation(
          {src_tensor}, creation_context_,
          absl::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(1, 1, 1, 4), &dst_tensor));

      std::vector<float> expected_output;
      if (diff == 0.0f) {
        expected_output.assign({0.0f, 0.0f, 0.0f, 0.0f});
      } else {
        const float ksqrt16 = std::sqrt(1.6f);
        const float ksqrt04 = std::sqrt(0.4f);
        expected_output.assign({-ksqrt16, -ksqrt04, ksqrt04, ksqrt16});
      }
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(tolerance), expected_output));
    }
  }
}

// note: 100.01 is not representable in FP16 (is in FP32), so use 101.0 instead.
INSTANTIATE_TEST_SUITE_P(
    uKernels, MeanStddevNormalizationTest,
    testing::Values(
        std::make_tuple(0.0f, 0.0f, 0.0f),         // zero mean, zero variance
        std::make_tuple(0.0f, 0.01f, 2.63e-4f),    // zero mean, small variance
        std::make_tuple(0.0f, 100.0f, 2.63e-4f),   // zero mean, large variance
        std::make_tuple(0.01f, 0.0f, 0.0f),        // small mean, zero variance
        std::make_tuple(0.01f, 0.01f, 3.57e-4f),   // small mean, small variance
        std::make_tuple(1.0f, 100.0f, 2.63e-4f),   // small mean, large variance
        std::make_tuple(100.0f, 0.0f, 0.0f),       // large mean, zero variance
        std::make_tuple(100.0f, 1.0f, 2.63e-4f),   // large mean, small variance
        std::make_tuple(100.0f, 100.0f, 2.63e-4f)  // large mean, large variance
        ));

TEST_F(OpenCLOperationTest, MeanStddevNormalizationAllBatches) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(9, 1, 1, 4);
  src_tensor.data = {
      0.0f,    0.0f,    0.0f,   0.0f,    // zero mean, zero variance
      -0.02f,  -0.01f,  0.01f,  0.02f,   // zero mean, small variance
      -200.0f, -100.0f, 100.0f, 200.0f,  // zero mean, large variance
      0.01f,   0.01f,   0.01f,  0.01f,   // small mean, zero variance
      -0.01f,  0.0f,    0.02f,  0.03f,   // small mean, small variance
      -199.0f, -99.0f,  101.0f, 201.0f,  // small mean, large variance
      100.0f,  100.0f,  100.0f, 100.0f,  // large mean, zero variance
      98.0f,   99.0f,   101.0f, 102.0f,  // large mean, small variance
      -100.0f, 0.0f,    200.0f, 300.0f,  // large mean, large variance
  };
  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps =
          precision == CalculationsPrecision::F32 ? 2.53e-05f : 3.57e-4f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation =
          CreateMeanStdDevNormalization(op_def, env_.GetDevicePtr()->info_, 1);
      ASSERT_OK(ExecuteGPUOperation(
          {src_tensor}, creation_context_,
          absl::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(9, 1, 1, 4), &dst_tensor));

      const float ksqrt16 = std::sqrt(1.6f);
      const float ksqrt04 = std::sqrt(0.4f);
      const std::vector<float> expected_output = {
          0.0f,     0.0f,     0.0f,    0.0f,     // zero mean, zero variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, small variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, large variance
          0.0f,     0.0f,     0.0f,    0.0f,     // small mean, zero variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, small variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, large variance
          0.0f,     0.0f,     0.0f,    0.0f,     // large mean, zero variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, small variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, large variance
      };
      EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_output))
          << "Failed using precision " << ToString(precision);
    }
  }
}

TEST_F(OpenCLOperationTest, MeanStddevNormalizationLargeVector) {
  const float mean = 100.0f;
  const float diff = 1.0f;
  // Some large vector that is not a round multiple of any SIMD vector sizes.
  constexpr int kVectorSize = 16 * 16 + 16 + 1;

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, kVectorSize);
  src_tensor.data.resize(kVectorSize);
  // First input is mean.
  src_tensor.data[0] = mean;
  // Rest is alternating between mean + diff and mean - diff.
  for (int i = 1; i < kVectorSize - 1; i += 2) {
    src_tensor.data[i + 0] = mean + diff;
    src_tensor.data[i + 1] = mean - diff;
  }

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps =
          precision == CalculationsPrecision::F32 ? 0.0f : 8.60e-4f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation = CreateMeanStdDevNormalization(
          op_def, env_.GetDevicePtr()->info_, (kVectorSize + 3) / 4);
      ASSERT_OK(ExecuteGPUOperation(
          {src_tensor}, creation_context_,
          absl::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(1, 1, 1, kVectorSize), &dst_tensor));

      float expected_output[kVectorSize];
      // First output should be 0.
      expected_output[0] = 0.0;
      // Rest should be alternating between ±√(N/(N-1)).
      const float expected_elem =
          std::sqrt(static_cast<double>(kVectorSize) /
                    static_cast<double>(kVectorSize - 1));
      for (int i = 1; i < kVectorSize - 1; i += 2) {
        expected_output[i + 0] = +expected_elem;
        expected_output[i + 1] = -expected_elem;
      }
      EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_output))
          << "Failed using precision " << ToString(precision);
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
