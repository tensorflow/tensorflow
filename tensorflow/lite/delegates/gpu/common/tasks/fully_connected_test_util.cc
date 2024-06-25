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

#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

absl::Status FullyConnectedTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 4);
  attr.weights.data = {0.0f, 1.0f, 2.0f, 3.0f,  //
                       4.0f, 5.0f, 6.0f, 7.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({14.5f, 37.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedLargeTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(12, 1, 1, 8);
  attr.weights.data = {
      0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,   //
      8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,  //
      16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,  //
      24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,  //
      32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,  //
      40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,  //
      48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,  //
      56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,  //
      64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f,  //
      72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,  //
      80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f,  //
      88.0f, 89.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f,  //
  };
  attr.bias.shape = Linear(12);
  attr.bias.data = {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f,
                    0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 0.0f : 1.0f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, 12), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({139.4f, 363.5f, 587.6f, 811.7f, 1035.8f, 1259.9f,
                         1484.1f, 1708.2f, 1932.3f, 2156.4f, 2380.5f, 2604.6f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedExtraLargeTest(TestExecutionEnvironment* env) {
  static const int kInputSize = 1024;
  static const int kOutputSize = 1024;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, kInputSize);
  src_tensor.data.assign(kInputSize, 1.1f);

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(1024, 1, 1, kInputSize);
  attr.weights.data.assign(kOutputSize * kInputSize, 2.2f);
  attr.bias.shape = Linear(kOutputSize);
  attr.bias.data.assign(kOutputSize, 3.3f);

  std::vector<float> expected(kOutputSize, 2481.38f);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      float eps;
      switch (precision) {
        case CalculationsPrecision::F32:
          eps = 2.45e-3f;
          break;
        case CalculationsPrecision::F32_F16:
          eps = 1.38f;
          break;
        case CalculationsPrecision::F16:
          eps = 39.0f;
          break;
      }
      if (precision == CalculationsPrecision::F32_F16 &&
          env->GetGpuInfo().IsApiMetal() && env->GetGpuInfo().IsIntel()) {
        eps = 3.5f;
      }
      if (precision == CalculationsPrecision::F32_F16 &&
          env->GetGpuInfo().IsGlsl()) {
        eps = 3.5f;
      }
      if (!env->GetGpuInfo().IsRoundToNearestSupported()) {
        eps *= 4.0f;
      }
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, kOutputSize), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  FullyConnectedInt8Attributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 4);
  attr.weights.data = {2,  4,  6,   8,  //
                       10, 12, -14, 16};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};
  attr.scale = 0.5f;
  attr.zero_point = 0;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({20.5f, 15.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
