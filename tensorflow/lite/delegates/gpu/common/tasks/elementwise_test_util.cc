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

#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise.h"

namespace tflite {
namespace gpu {

absl::Status AbsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ABS);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.0f), half(1.0f), half(0.05f), half(0.045f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status CosTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -0.05f, 0.045f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 5e-5f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::cos(0.0f), std::cos(-1.0f), std::cos(-0.05f), std::cos(0.045f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status CopyTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COPY);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(src_tensor.data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status EluTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ELU);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 7), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {0.0f, 1.0f, std::exp(-1.0f) - 1.0f, 100.0f, std::exp(-100.0f) - 1.0f,
           0.01f, std::exp(-0.01f) - 1.0f},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ExpTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {0.0f, 1.0f, -1.0f, 2.5f, -1.7f, 0.01f, -0.01f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 2e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::EXP);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 7), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f), std::exp(2.5f),
           std::exp(-1.7f), std::exp(0.01f), std::exp(-0.01f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FloorTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::FLOOR);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {-5.0, -3.0f, -2.0f, 0.0f, 1.0f, 3.0f, 4.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FloorDivTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::FLOOR_DIV, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({std::floor(-4.5f / scalar), std::floor(-3.0f / scalar),
                         std::floor(-1.5f / scalar), std::floor(0.0f / scalar),
                         std::floor(1.5f / scalar), std::floor(3.0f / scalar),
                         std::floor(4.5f / scalar)},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FloorModTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::FLOOR_MOD, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({-4.5f - std::floor(-4.5f / scalar) * scalar,
                         -3.0f - std::floor(-3.0f / scalar) * scalar,
                         -1.5f - std::floor(-1.5f / scalar) * scalar,
                         0.0f - std::floor(0.0f / scalar) * scalar,
                         1.5f - std::floor(1.5f / scalar) * scalar,
                         3.0f - std::floor(3.0f / scalar) * scalar,
                         4.5f - std::floor(4.5f / scalar) * scalar},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status HardSwishTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::HARD_SWISH);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, 0.0f, -0.375f, 0.0f, 1.125f, 3.f, 4.5f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status LogTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::LOG);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::log(1.0f), std::log(2.0f), std::log(3.0f), std::log(4.0f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status NegTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, -2.0f, 0.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::NEG);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({-1.0f, 2.0f, 0.0f, -4.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status RsqrtTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::RSQRT);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f / std::sqrt(1.0f), 1.0f / std::sqrt(2.0f),
                         1.0f / std::sqrt(3.0f), 1.0f / std::sqrt(4.0f)},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SigmoidTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-std::log(1.0f), -std::log(2.0f), -std::log(3.0f),
                     -std::log(4.0f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIGMOID);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({0.5f, 1.0f / 3.0f, 0.25f, 0.2f},
                                    dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SinTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -0.05f, 0.045f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 2e-5f : 5e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIN);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::sin(0.0f), std::sin(-1.0f), std::sin(-0.05f), std::sin(0.045f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SqrtTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SQRT);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f), std::sqrt(4.0f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SquareTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, -2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SQUARE);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 4.0f, 9.0f, 16.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status TanhTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-4.0f, -0.1f, 0.1f, 2.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({std::tanh(-4.0f), std::tanh(-0.1f),
                                     std::tanh(0.1f), std::tanh(2.0f)},
                                    dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SubTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::SUB, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 1.0f, 0.0f, 0.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SquaredDiffTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::SQUARED_DIFF, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.25f, 1.0f, 0.0f, 0.25f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status DivTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::DIV, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({2.0f, 2.0f, 1.0f, 3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status PowTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {6.0f, 7.0f, 4.0f, 2.0f};
  src_tensor_1.data = {0.0f, 1.0f, 2.0f, 3.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::POW, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 7.0f, 16.0f, 8.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status AddTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::ADD, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.5f, 3.0f, 6.0f, 6.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};
  src_tensor_1.data = {1.0f, 2.0f, 3.0f, -2.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MAXIMUM, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 2.0f, 3.0f, -2.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumWithScalarTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -1.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, -1.0f, 2.0f, -1.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumWithConstantLinearTensorTest(
    TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> linear_tensor;
  linear_tensor.shape = Linear(2);
  linear_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = linear_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 2.0f, 0.5f, 3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumWithConstantHWCTensorTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  ::tflite::gpu::Tensor<HWC, DataType::FLOAT32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 2);
  hwc_tensor.data = {0.5f, 2.0f, 0.7f, 4.7f};
  ElementwiseAttributes attr;
  attr.param = hwc_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 2.0f, 0.7f, 4.7f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}
absl::Status MaximumWithConstantHWCTensorBroadcastChannelsTest(
    TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  ::tflite::gpu::Tensor<HWC, DataType::FLOAT32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 1);
  hwc_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = hwc_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 0.5f, 2.0f, 3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MinimumTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};
  src_tensor_1.data = {1.0f, 2.0f, 3.0f, -2.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MINIMUM, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, -6.2f, 2.0f, -3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MinimumWithScalarTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -1.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MINIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({-1.0f, -6.2f, -1.0f, -3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MUL, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 2.0f, 9.0f, 6.75f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulBroadcastHWTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 1, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 3.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MUL, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 6.0f, 1.5f, 13.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulBroadcastChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 3.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MUL, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 1.0f, 9.0f, 13.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SubWithScalarAtFirstPositionTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = 4.0f;
  attr.runtime_tensor_is_second = true;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::SUB, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({4.0f, 10.2f, 2.0f, 7.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status LessTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};
  src_tensor_1.data = {1.0f, 0.0f, 2.0f, -4.0f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, false, false, false};

  for (auto src_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});

      TensorDescriptor src_desc0, src_desc1, dst_desc;
      src_desc0 = op_def.src_tensors[0];
      src_desc0.UploadData(src_tensor_0);
      src_desc1 = op_def.src_tensors[1];
      src_desc1.UploadData(src_tensor_1);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::LESS, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc0, &src_desc1}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LessEqualTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, true, true, false};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto src_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src_tensor_0);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::LESS_EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status GreaterTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, false, true};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto src_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src_tensor_0);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::GREATER, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status GreaterEqualTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, true, true};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto src_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src_tensor_0);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::GREATER_EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status EqualTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, true, false};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto src_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src_tensor_0);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status NotEqualTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, true, false, true};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto src_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src_tensor_0);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::NOT_EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CosBroadcastTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 1);
  src_tensor.data = {0.7f, -1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 5e-5f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      BHWC output_shape(1, 2, 1, 2);
      GPUOperation operation = CreateElementwiseOneInputWithBroadcast(
          env->GetGpuInfo(), op_def, OperationType::COS, src_tensor.shape,
          output_shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          output_shape, &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::cos(0.7f), std::cos(0.7f), std::cos(-1.5f), std::cos(-1.5f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumScalarBroadcastInputTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -2.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      BHWC output_shape(1, 2, 1, 2);
      GPUOperation operation = CreateElementwiseWithBroadcast(
          env->GetGpuInfo(), op_def, OperationType::MAXIMUM, attr,
          src_tensor_0.shape, output_shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          output_shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({2.0f, 2.0f, -2.0f, -2.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulLinearBroadcastInputTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {2.0f, -3.0f};

  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> linear_tensor;
  linear_tensor.shape = Linear(2);
  linear_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = linear_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      BHWC output_shape(1, 2, 1, 2);
      GPUOperation operation = CreateElementwiseWithBroadcast(
          env->GetGpuInfo(), op_def, OperationType::MUL, attr,
          src_tensor_0.shape, output_shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
          output_shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 4.0f, -1.5f, -6.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulBroadcastBothInputsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 1, 2, 1);
  src_tensor_1.shape = BHWC(1, 1, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f};
  src_tensor_1.data = {3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      BHWC output_shape(1, 1, 2, 2);
      GPUOperation operation = CreateElementwiseTwoInputWithBroadcast(
          op_def, OperationType::MUL, src_tensor_0.shape, src_tensor_1.shape,
          output_shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          std::make_unique<GPUOperation>(std::move(operation)), output_shape,
          &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({3.0f, 4.0f, 6.0f, 8.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status LogicalAndTest(TestExecutionEnvironment* env) {
  using TensorBool32 = Tensor<BHWC, DataType::BOOL>;
  TensorBool32 src_tensor_0, src_tensor_1, ref_tensor;
  src_tensor_0.shape = BHWC(1, 1, 2, 24);
  src_tensor_1.shape = BHWC(1, 1, 2, 24);
  ref_tensor.shape = BHWC(1, 1, 2, 24);
  for (int i = 0; i < 48; i++) {
    bool value = i % 2 == 0;
    src_tensor_0.data.push_back(value);
    src_tensor_1.data.push_back(i < 24 ? value : !value);
    ref_tensor.data.push_back(i < 24 ? value : false);
  }

  for (auto src_storage : env->GetSupportedStorages(DataType::BOOL)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back({DataType::BOOL, src_storage, Layout::HWC});
      op_def.src_tensors.push_back({DataType::BOOL, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});

      TensorDescriptor src_desc0, src_desc1, dst_desc;
      src_desc0 = op_def.src_tensors[0];
      src_desc0.UploadData(src_tensor_0);
      src_desc1 = op_def.src_tensors[1];
      src_desc1.UploadData(src_tensor_1);
      dst_desc.SetBHWCShape(BHWC(1, 1, 2, 24));

      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::LOGICAL_AND, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc0, &src_desc1}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LogicalAndWithConstantTest(TestExecutionEnvironment* env) {
  using TensorBool32 = Tensor<BHWC, DataType::BOOL>;
  TensorBool32 src_tensor_0, ref_tensor;
  src_tensor_0.shape = BHWC(1, 1, 2, 24);
  ref_tensor.shape = BHWC(1, 1, 2, 24);

  Tensor<HWC, DataType::BOOL> src_tensor_1;
  src_tensor_1.shape = HWC(1, 2, 24);
  for (int i = 0; i < 48; i++) {
    bool value = i % 2 == 0;
    src_tensor_0.data.push_back(value);
    src_tensor_1.data.push_back(i < 24 ? value : !value);
    ref_tensor.data.push_back(i < 24 ? value : false);
  }

  ElementwiseAttributesBase<DataType::BOOL, bool> attr;
  attr.param = src_tensor_1;

  for (auto src_storage : env->GetSupportedStorages(DataType::BOOL)) {
    for (auto dst_storage : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back({DataType::BOOL, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});

      TensorDescriptor src_desc0, src_desc1, dst_desc;
      src_desc0 = op_def.src_tensors[0];
      src_desc0.UploadData(src_tensor_0);
      dst_desc.SetBHWCShape(BHWC(1, 1, 2, 24));

      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::LOGICAL_AND, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc0}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::BOOL> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
