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

#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"

#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

absl::Status ClExecutionEnvironment::Init() { return CreateEnvironment(&env_); }

std::vector<CalculationsPrecision>
ClExecutionEnvironment::GetSupportedPrecisions() const {
  return env_.GetSupportedPrecisions();
}

std::vector<TensorStorageType> ClExecutionEnvironment::GetSupportedStorages(
    DataType data_type) const {
  return env_.GetSupportedStorages();
}

std::vector<TensorStorageType>
ClExecutionEnvironment::GetSupportedStoragesWithHWZeroClampSupport(
    DataType data_type) const {
  return env_.GetSupportedStoragesWithHWZeroClampSupport();
}

const GpuInfo& ClExecutionEnvironment::GetGpuInfo() const {
  return env_.GetDevicePtr()->GetInfo();
}

absl::Status ClExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorFloat32*>& dst_cpu) {
  CreationContext creation_context;
  creation_context.device = env_.GetDevicePtr();
  creation_context.context = &env_.context();
  creation_context.queue = env_.queue();
  creation_context.cache = env_.program_cache();

  const OperationDef& op_def = operation->GetDefinition();
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(creation_context.queue, src_cpu[i]));
    operation->SetSrc(&src[i], i);
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));

    operation->SetDst(&dst[i], i);
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  RETURN_IF_ERROR(cl_op.Compile(creation_context));
  RETURN_IF_ERROR(cl_op.UpdateParams());
  RETURN_IF_ERROR(cl_op.AddToQueue(creation_context.queue));
  RETURN_IF_ERROR(creation_context.queue->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(creation_context.queue, dst_cpu[i]));
  }
  return absl::OkStatus();
}

absl::Status ClExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<Tensor5DFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWDC>& dst_sizes,
    const std::vector<Tensor5DFloat32*>& dst_cpu) {
  CreationContext creation_context;
  creation_context.device = env_.GetDevicePtr();
  creation_context.context = &env_.context();
  creation_context.queue = env_.queue();
  creation_context.cache = env_.program_cache();

  const OperationDef& op_def = operation->GetDefinition();
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(creation_context.queue, src_cpu[i]));
    operation->SetSrc(&src[i], i);
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));

    operation->SetDst(&dst[i], i);
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  RETURN_IF_ERROR(cl_op.Compile(creation_context));
  RETURN_IF_ERROR(cl_op.UpdateParams());
  RETURN_IF_ERROR(cl_op.AddToQueue(creation_context.queue));
  RETURN_IF_ERROR(creation_context.queue->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(creation_context.queue, dst_cpu[i]));
  }
  return absl::OkStatus();
}

absl::Status ClExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  CreationContext creation_context;
  creation_context.device = env_.GetDevicePtr();
  creation_context.context = &env_.context();
  creation_context.queue = env_.queue();
  creation_context.cache = env_.program_cache();

  const OperationDef& op_def = operation->GetDefinition();
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i]->GetBHWDCShape();
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(
        src[i].CreateFromDescriptor(*src_cpu[i], creation_context.context));
    operation->SetSrc(&src[i], i);
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_cpu[i]->GetBHWDCShape();
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));

    operation->SetDst(&dst[i], i);
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  RETURN_IF_ERROR(cl_op.Compile(creation_context));
  RETURN_IF_ERROR(cl_op.UpdateParams());
  RETURN_IF_ERROR(cl_op.AddToQueue(creation_context.queue));
  RETURN_IF_ERROR(creation_context.queue->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    RETURN_IF_ERROR(dst[i].ToDescriptor(dst_cpu[i], creation_context.queue));
  }
  return absl::OkStatus();
}

absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                 const CreationContext& creation_context,
                                 std::unique_ptr<GPUOperation>&& operation,
                                 const std::vector<BHWC>& dst_sizes,
                                 const std::vector<TensorFloat32*>& dst_cpu) {
  const OperationDef& op_def = operation->GetDefinition();
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(creation_context.queue, src_cpu[i]));
    operation->SetSrc(&src[i], i);
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(*creation_context.context, dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));

    operation->SetDst(&dst[i], i);
  }
  RETURN_IF_ERROR(operation->AssembleCode(creation_context.device->GetInfo()));

  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  RETURN_IF_ERROR(cl_op.Compile(creation_context));
  RETURN_IF_ERROR(cl_op.UpdateParams());
  RETURN_IF_ERROR(cl_op.AddToQueue(creation_context.queue));
  RETURN_IF_ERROR(creation_context.queue->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(creation_context.queue, dst_cpu[i]));
  }
  return absl::OkStatus();
}

absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                 const CreationContext& creation_context,
                                 std::unique_ptr<GPUOperation>&& operation,
                                 const BHWC& dst_size, TensorFloat32* result) {
  return ExecuteGPUOperation(std::vector<TensorFloat32>{src_cpu},
                             creation_context, std::move(operation),
                             std::vector<BHWC>{dst_size},
                             std::vector<TensorFloat32*>{result});
}

absl::Status ExecuteGPUOperation(const TensorFloat32& src_cpu,
                                 const CreationContext& creation_context,
                                 std::unique_ptr<GPUOperation>&& operation,
                                 const BHWC& dst_size, TensorFloat32* result) {
  return ExecuteGPUOperation(std::vector<TensorFloat32>{src_cpu},
                             creation_context, std::move(operation), dst_size,
                             result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
