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

#include "tensorflow/lite/delegates/gpu/common/tasks/softmax_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.h"

namespace tflite {
namespace gpu {

absl::Status SoftmaxTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {std::log(1.0f), std::log(2.0f), std::log(3.0f),
                     std::log(4.0f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSoftmax(op_def);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 7.0f, 4.0f / 7.0f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SoftmaxBigNumberTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  src_tensor.data.resize(4);
  src_tensor.data[0] = doubles[0];
  src_tensor.data[1] = doubles[1];
  src_tensor.data[2] = doubles[2];
  src_tensor.data[3] = doubles[3];
  if (!std::isinf(std::exp(src_tensor.data[3]))) {
    return absl::InternalError("exp(100.0f) not inf in float (32 bit)");
  }
  if (std::isinf(std::exp(doubles[3]))) {
    return absl::InternalError("exp(100.0) inf in double (64 bit)");
  }
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]);
  double s1 = std::exp(doubles[2]) + std::exp(doubles[3]);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSoftmax(op_def);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({static_cast<float>(std::exp(doubles[0]) / s0),
                         static_cast<float>(std::exp(doubles[1]) / s0),
                         static_cast<float>(std::exp(doubles[2]) / s1),
                         static_cast<float>(std::exp(doubles[3]) / s1)},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status Softmax1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {std::log(1.0f), std::log(2.0f), std::log(3.0f),
                     std::log(4.0f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Softmax1x1 operation = CreateSoftmax1x1(op_def);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Softmax1x1>(std::move(operation)),
          BHWC(1, 1, 1, 4), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.1f, 0.2f, 0.3f, 0.4f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status Softmax1x1BigNumberTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  src_tensor.data.resize(4);
  src_tensor.data[0] = doubles[0];
  src_tensor.data[1] = doubles[1];
  src_tensor.data[2] = doubles[2];
  src_tensor.data[3] = doubles[3];
  if (!std::isinf(std::exp(src_tensor.data[3]))) {
    return absl::InternalError("exp(100.0f) not inf in float (32 bit)");
  }
  if (std::isinf(std::exp(doubles[3]))) {
    return absl::InternalError("exp(100.0) inf in double (64 bit)");
  }
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]) +
              std::exp(doubles[2]) + std::exp(doubles[3]);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Softmax1x1 operation = CreateSoftmax1x1(op_def);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Softmax1x1>(std::move(operation)),
          BHWC(1, 1, 1, 4), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({static_cast<float>(std::exp(doubles[0]) / s0),
                         static_cast<float>(std::exp(doubles[1]) / s0),
                         static_cast<float>(std::exp(doubles[2]) / s0),
                         static_cast<float>(std::exp(doubles[3]) / s0)},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
