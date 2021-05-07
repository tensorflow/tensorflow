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

#include "tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reduce.h"

namespace tflite {
namespace gpu {

absl::Status MeanHWTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::set<tflite::gpu::Axis> axis{Axis::HEIGHT, Axis::WIDTH};

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::MEAN, op_def,
                       env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 1, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceSumChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 5);
  src_tensor.data = {1.1, 2.1, 0.7, 0.3, 1.2, 3.1, 4.1, 0.0, 1.0, 4.4};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_SUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({5.4f, 12.6f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceProductChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.1, 2.0, 3.1, 4.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_PRODUCT,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2.2f, 12.4f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceMaxChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3, -100.0, 32.6, 1.1,
                     -3.1, -4.0, -5.0, -7.0,   -2.0, -100.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MAXIMUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({32.6f, -2.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceMinChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3, -100.0, 32.6, 1.1,
                     -3.1, -4.0, -5.0, -7.0,   -2.0, 100.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MINIMUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({-100.0f, -7.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
