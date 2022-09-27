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

#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {

absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  const OperationDef& op_def = operation->GetDefinition();
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i]->GetBHWDCShape();
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
  }

  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_cpu[i]->GetBHWDCShape();
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));
  return ExecuteGpuOperationInternal(src_cpu, dst_cpu, std::move(operation));
}

absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorFloat32*>& dst_cpu) {
  const OperationDef& op_def = operation->GetDefinition();
  std::vector<TensorDescriptor> src_cpu_descs(src_cpu.size());
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    src_cpu_descs[i] = op_def.src_tensors[i];
    src_cpu_descs[i].UploadData(src_cpu[i]);
    src_cpu_desc_ptrs[i] = &src_cpu_descs[i];
  }
  std::vector<TensorDescriptor> dst_cpu_descs(dst_cpu.size());
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu_descs[i] = op_def.dst_tensors[i];
    dst_cpu_descs[i].SetBHWCShape(dst_sizes[i]);
    dst_cpu_desc_ptrs[i] = &dst_cpu_descs[i];
  }

  RETURN_IF_ERROR(ExecuteGPUOperation(src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
                                      std::move(operation)));

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu_descs[i].DownloadData(dst_cpu[i]);
  }
  return absl::OkStatus();
}

absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<Tensor5DFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWDC>& dst_sizes,
    const std::vector<Tensor5DFloat32*>& dst_cpu) {
  const OperationDef& op_def = operation->GetDefinition();
  std::vector<TensorDescriptor> src_cpu_descs(src_cpu.size());
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    src_cpu_descs[i] = op_def.src_tensors[i];
    src_cpu_descs[i].UploadData(src_cpu[i]);
    src_cpu_desc_ptrs[i] = &src_cpu_descs[i];
  }
  std::vector<TensorDescriptor> dst_cpu_descs(dst_cpu.size());
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu_descs[i] = op_def.dst_tensors[i];
    dst_cpu_descs[i].SetBHWDCShape(dst_sizes[i]);
    dst_cpu_desc_ptrs[i] = &dst_cpu_descs[i];
  }

  RETURN_IF_ERROR(ExecuteGPUOperation(src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
                                      std::move(operation)));

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu_descs[i].DownloadData(dst_cpu[i]);
  }
  return absl::OkStatus();
}

absl::Status PointWiseNear(const std::vector<float>& ref,
                           const std::vector<float>& to_compare, float eps) {
  if (ref.size() != to_compare.size()) {
    return absl::InternalError(absl::StrCat("ref size(", ref.size(),
                                            ") != to_compare size(",
                                            to_compare.size(), ")"));
  }
  for (int i = 0; i < ref.size(); ++i) {
    const float abs_diff = fabs(ref[i] - to_compare[i]);
    if (abs_diff > eps) {
      return absl::InternalError(absl::StrCat(
          "ref[", i, "] = ", ref[i], ", to_compare[", i, "] = ", to_compare[i],
          ", abs diff = ", abs_diff, " > ", eps, " (eps)"));
    }
  }
  return absl::OkStatus();
}

absl::Status TestExecutionEnvironment::ExecuteGpuModel(
    const std::vector<TensorFloat32>& src_cpu,
    const std::vector<TensorFloat32*>& dst_cpu, GpuModel* gpu_model) {
  std::vector<TensorFloat32> inputs = src_cpu;
  for (int k = 0; k < gpu_model->nodes.size(); ++k) {
    auto& gpu_node = gpu_model->nodes[k];
    std::vector<TensorDescriptor> src_cpu_descs(gpu_node.inputs.size());
    std::vector<TensorDescriptor*> src_cpu_desc_ptrs(gpu_node.inputs.size());
    for (int i = 0; i < gpu_node.inputs.size(); ++i) {
      src_cpu_descs[i] = gpu_model->tensors[gpu_node.inputs[i]];
      src_cpu_descs[i].UploadData(inputs[i]);
      src_cpu_desc_ptrs[i] = &src_cpu_descs[i];
    }

    std::vector<TensorDescriptor> dst_cpu_descs(gpu_node.outputs.size());
    std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(gpu_node.outputs.size());
    for (int i = 0; i < gpu_node.outputs.size(); ++i) {
      dst_cpu_descs[i] = gpu_model->tensors[gpu_node.outputs[i]];
      dst_cpu_desc_ptrs[i] = &dst_cpu_descs[i];
    }

    RETURN_IF_ERROR(ExecuteGpuOperationInternal(
        src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
        std::move(gpu_model->nodes[k].gpu_operation)));

    inputs.resize(gpu_node.outputs.size());
    for (int i = 0; i < gpu_node.outputs.size(); ++i) {
      dst_cpu_descs[i].DownloadData(&inputs[i]);
    }
  }
  if (dst_cpu.size() != inputs.size()) {
    return absl::InternalError("Size mismatch.");
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    *dst_cpu[i] = inputs[i];
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
