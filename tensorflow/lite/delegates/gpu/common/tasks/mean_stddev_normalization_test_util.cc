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

#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.h"

#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"

namespace tflite {
namespace gpu {

// Parameterized test: mean, difference, tolerance.
// Input is constructed as [mean-2*diff, mean-diff, mean+diff, mean+2*diff]
absl::Status MeanStddevNormSeparateBatchesTest(float mean, float diff,
                                               float tolerance,
                                               TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 4);
  src_tensor.data = {mean - 2 * diff, mean - diff,     mean + diff,
                     mean + 2 * diff, mean - 2 * diff, mean - diff,
                     mean + diff,     mean + 2 * diff};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      auto operation = CreateMeanStdDevNormalization(op_def, env->GetGpuInfo(),
                                                     src_tensor.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor},
          std::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(1, 1, 2, 4), &dst_tensor));

      std::vector<float> expected_output;
      if (diff == 0.0f) {
        expected_output.assign(
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
      } else {
        const float ksqrt16 = std::sqrt(1.6f);
        const float ksqrt04 = std::sqrt(0.4f);
        expected_output.assign({-ksqrt16, -ksqrt04, ksqrt04, ksqrt16, -ksqrt16,
                                -ksqrt04, ksqrt04, ksqrt16});
      }
      RETURN_IF_ERROR(
          PointWiseNear(expected_output, dst_tensor.data, tolerance));

      TensorFloat32 dst_tensor_single_step;
      auto operation_single_step = CreateMeanStdDevNormalization(
          op_def, env->GetGpuInfo(), src_tensor.shape,
          /*variance_bias*/ 1.0e-8f, /*two_step*/ false);
      RETURN_IF_ERROR(
          env->ExecuteGPUOperation({src_tensor},
                                   std::make_unique<MeanStdDevNormalization>(
                                       std::move(operation_single_step)),
                                   BHWC(1, 1, 2, 4), &dst_tensor_single_step));
      RETURN_IF_ERROR(PointWiseNear(expected_output,
                                    dst_tensor_single_step.data, tolerance));
    }
  }
  return absl::OkStatus();
}

absl::Status MeanStddevNormalizationAllBatchesTest(
    TestExecutionEnvironment* env) {
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
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps =
          precision == CalculationsPrecision::F32 ? 2.53e-05f : 3.57e-4f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation = CreateMeanStdDevNormalization(op_def, env->GetGpuInfo(),
                                                     src_tensor.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor},
          std::make_unique<MeanStdDevNormalization>(std::move(operation)),
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
      RETURN_IF_ERROR(PointWiseNear(expected_output, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);

      TensorFloat32 dst_tensor_single_step;
      auto operation_single_step = CreateMeanStdDevNormalization(
          op_def, env->GetGpuInfo(), src_tensor.shape,
          /*variance_bias*/ 1.0e-8f, /*two_step*/ false);
      RETURN_IF_ERROR(
          env->ExecuteGPUOperation({src_tensor},
                                   std::make_unique<MeanStdDevNormalization>(
                                       std::move(operation_single_step)),
                                   BHWC(9, 1, 1, 4), &dst_tensor_single_step));
      RETURN_IF_ERROR(
          PointWiseNear(expected_output, dst_tensor_single_step.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status MeanStddevNormalizationLargeVectorTest(
    TestExecutionEnvironment* env) {
  const float mean = 100.0f;
  const float diff = 1.0f;
  // Some large vector that is not a round multiple of any SIMD vector sizes.
  constexpr int kVectorSize = 16 * 16 + 16 + 1;

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, kVectorSize);
  src_tensor.data.resize(kVectorSize * 2);
  // First input is mean.
  src_tensor.data[0] = mean;
  src_tensor.data[kVectorSize] = mean;
  // Rest is alternating between mean + diff and mean - diff.
  for (int i = 1; i < kVectorSize - 1; i += 2) {
    src_tensor.data[i + 0] = mean + diff;
    src_tensor.data[i + 1] = mean - diff;
    src_tensor.data[kVectorSize + i + 0] = mean + diff;
    src_tensor.data[kVectorSize + i + 1] = mean - diff;
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps =
          precision == CalculationsPrecision::F32 ? 0.0f : 8.60e-4f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      auto operation = CreateMeanStdDevNormalization(op_def, env->GetGpuInfo(),
                                                     src_tensor.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor},
          std::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(1, 1, 2, kVectorSize), &dst_tensor));

      std::vector<float> expected_output(kVectorSize * 2);
      // First output should be 0.
      expected_output[0] = 0.0;
      expected_output[kVectorSize] = 0.0;
      // Rest should be alternating between ±√(N/(N-1)).
      const float expected_elem =
          std::sqrt(static_cast<double>(kVectorSize) /
                    static_cast<double>(kVectorSize - 1));
      for (int i = 1; i < kVectorSize - 1; i += 2) {
        expected_output[i + 0] = +expected_elem;
        expected_output[i + 1] = -expected_elem;
        expected_output[kVectorSize + i + 0] = +expected_elem;
        expected_output[kVectorSize + i + 1] = -expected_elem;
      }
      RETURN_IF_ERROR(PointWiseNear(expected_output, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);

      if (precision != CalculationsPrecision::F32) {
        TensorFloat32 dst_tensor_single_step;
        auto operation_single_step = CreateMeanStdDevNormalization(
            op_def, env->GetGpuInfo(), src_tensor.shape,
            /*variance_bias*/ 1.0e-8f, /*two_step*/ false);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {src_tensor},
            std::make_unique<MeanStdDevNormalization>(
                std::move(operation_single_step)),
            BHWC(1, 1, 2, kVectorSize), &dst_tensor_single_step));
        RETURN_IF_ERROR(
            PointWiseNear(expected_output, dst_tensor_single_step.data, eps))
            << "Failed using precision " << ToString(precision);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
