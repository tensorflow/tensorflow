/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/cast_test_util.h"

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/cast.h"

namespace tflite {
namespace gpu {

absl::Status CastTests(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, DataType::FLOAT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {0.0f, -1.3f, -7.4f, 12.45f};

  tflite::gpu::Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {0, -1, -7, 12};

  for (auto storage_src : env->GetSupportedStorages(DataType::FLOAT16)) {
    for (auto storage_dst : env->GetSupportedStorages(DataType::INT32)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F16;
      op_def.src_tensors.push_back(
          {DataType::FLOAT16, storage_src, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::INT32, storage_dst, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateCast(op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::INT32> dst_tensor;
      dst_desc.DownloadData(&dst_tensor);
      if (dst_tensor.data != ref_tensor.data) {
        return absl::InternalError("not equal");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CastToBoolTests(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, DataType::FLOAT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {0.0f, -1.3f, -7.4f, 12.45f};

  tflite::gpu::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, true, true, true};

  for (auto storage_src : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto storage_dst : env->GetSupportedStorages(DataType::BOOL)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back(
          {DataType::FLOAT32, storage_src, Layout::HWC});
      op_def.dst_tensors.push_back({DataType::BOOL, storage_dst, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateCast(op_def, env->GetGpuInfo());
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

absl::Status CastFromBoolTests(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, DataType::BOOL> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {false, true, true, true};

  tflite::gpu::Tensor<BHWC, DataType::FLOAT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {0.0, 1.0, 1.0, 1.0};

  for (auto storage_src : env->GetSupportedStorages(DataType::BOOL)) {
    for (auto storage_dst : env->GetSupportedStorages(DataType::FLOAT32)) {
      OperationDef op_def;
      op_def.precision = CalculationsPrecision::F32;
      op_def.src_tensors.push_back({DataType::BOOL, storage_src, Layout::HWC});
      op_def.dst_tensors.push_back(
          {DataType::FLOAT32, storage_dst, Layout::HWC});
      TensorDescriptor src_desc, dst_desc;
      src_desc = op_def.src_tensors[0];
      src_desc.UploadData(src);
      dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
      GPUOperation operation = CreateCast(op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_desc}, {&dst_desc},
          std::make_unique<GPUOperation>(std::move(operation))));

      tflite::gpu::Tensor<BHWC, DataType::FLOAT32> dst_tensor;
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
