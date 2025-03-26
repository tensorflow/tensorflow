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

#include "tensorflow/lite/delegates/gpu/common/tasks/gather_test_util.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/gather.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

absl::Status GatherBatchTest(TestExecutionEnvironment* env, bool constant_idx) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(5, 1, 1, 1);
  src_tensor.data = {half(1.5f), half(2.4f), half(3.3f), half(4.2f),
                     half(5.1f)};
  std::vector<int> src_indices_data{1, 2, 3, 0, 1, 4, 2, 3, 1};
  GatherAttributes attr;
  attr.axis = Axis::BATCH;
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorDescriptor src_0, src_1, dst;
      src_0 = op_def.src_tensors[0];
      src_0.UploadData(src_tensor);
      dst.SetBHWDCShape(BHWDC(9, 1, 1, 1, 1));
      if (!constant_idx) {
        op_def.src_tensors.push_back({DataType::INT32, storage, Layout::BHWC});
        TensorDescriptor src_1;
        src_1 = op_def.src_tensors[1];
        tflite::gpu::Tensor<BHWC, DataType::INT32> src_indices;
        src_indices.shape = BHWC(9, 1, 1, 1);
        src_indices.data = src_indices_data;
        src_1.UploadData(src_indices);
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0, &src_1}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      } else {
        attr.indices.shape = Linear(9);
        attr.indices.data = src_indices_data;
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      }
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(PointWiseNear(
          {half(2.4f), half(3.3f), half(4.2f), half(1.5f), half(2.4f),
           half(5.1f), half(3.3f), half(4.2f), half(2.4f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status GatherHeightTest(TestExecutionEnvironment* env,
                              bool constant_idx) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 5, 1, 1);
  src_tensor.data = {half(1.5f), half(2.4f), half(3.3f), half(4.2f),
                     half(5.1f)};
  std::vector<int> src_indices_data{1, 2, 3, 0, 1, 4, 2, 3, 1};
  GatherAttributes attr;
  attr.axis = Axis::HEIGHT;
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorDescriptor src_0, src_1, dst;
      src_0 = op_def.src_tensors[0];
      src_0.UploadData(src_tensor);
      dst.SetBHWDCShape(BHWDC(1, 9, 1, 1, 1));
      if (!constant_idx) {
        op_def.src_tensors.push_back({DataType::INT32, storage, Layout::BHWC});
        TensorDescriptor src_1;
        src_1 = op_def.src_tensors[1];
        tflite::gpu::Tensor<BHWC, DataType::INT32> src_indices;
        src_indices.shape = BHWC(9, 1, 1, 1);
        src_indices.data = src_indices_data;
        src_1.UploadData(src_indices);
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0, &src_1}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      } else {
        attr.indices.shape = Linear(9);
        attr.indices.data = src_indices_data;
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      }
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(PointWiseNear(
          {half(2.4f), half(3.3f), half(4.2f), half(1.5f), half(2.4f),
           half(5.1f), half(3.3f), half(4.2f), half(2.4f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status GatherWidthTest(TestExecutionEnvironment* env, bool constant_idx) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 5, 1);
  src_tensor.data = {half(1.5f), half(2.4f), half(3.3f), half(4.2f),
                     half(5.1f)};
  std::vector<int> src_indices_data{1, 2, 3, 0, 1, 4, 2, 3, 1};
  GatherAttributes attr;
  attr.axis = Axis::WIDTH;
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorDescriptor src_0, src_1, dst;
      src_0 = op_def.src_tensors[0];
      src_0.UploadData(src_tensor);
      dst.SetBHWDCShape(BHWDC(1, 1, 9, 1, 1));
      GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
      if (!constant_idx) {
        op_def.src_tensors.push_back({DataType::INT32, storage, Layout::BHWC});
        TensorDescriptor src_1;
        src_1 = op_def.src_tensors[1];
        tflite::gpu::Tensor<BHWC, DataType::INT32> src_indices;
        src_indices.shape = BHWC(9, 1, 1, 1);
        src_indices.data = src_indices_data;
        src_1.UploadData(src_indices);
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0, &src_1}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      } else {
        attr.indices.shape = Linear(9);
        attr.indices.data = src_indices_data;
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      }
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(PointWiseNear(
          {half(2.4f), half(3.3f), half(4.2f), half(1.5f), half(2.4f),
           half(5.1f), half(3.3f), half(4.2f), half(2.4f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status GatherChannelsTest(TestExecutionEnvironment* env,
                                bool constant_idx) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 5);
  src_tensor.data = {half(1.5f), half(2.4f), half(3.3f), half(4.2f),
                     half(5.1f)};
  std::vector<int> src_indices_data{1, 2, 3, 0, 1, 4, 2, 3, 1};
  GatherAttributes attr;
  attr.axis = Axis::CHANNELS;
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorDescriptor src_0, src_1, dst;
      src_0 = op_def.src_tensors[0];
      src_0.UploadData(src_tensor);
      dst.SetBHWDCShape(BHWDC(1, 1, 1, 1, 9));
      GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
      if (!constant_idx) {
        op_def.src_tensors.push_back({DataType::INT32, storage, Layout::BHWC});
        TensorDescriptor src_1;
        src_1 = op_def.src_tensors[1];
        tflite::gpu::Tensor<BHWC, DataType::INT32> src_indices;
        src_indices.shape = BHWC(9, 1, 1, 1);
        src_indices.data = src_indices_data;
        src_1.UploadData(src_indices);
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0, &src_1}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      } else {
        attr.indices.shape = Linear(9);
        attr.indices.data = src_indices_data;
        GPUOperation operation = CreateGather(env->GetGpuInfo(), op_def, attr);
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            {&src_0}, {&dst},
            std::make_unique<GPUOperation>(std::move(operation))));
      }
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(PointWiseNear(
          {half(2.4f), half(3.3f), half(4.2f), half(1.5f), half(2.4f),
           half(5.1f), half(3.3f), half(4.2f), half(2.4f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
