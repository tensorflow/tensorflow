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

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
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
  if (data_type == DataType::FLOAT16 &&
      !env_.GetDevicePtr()->GetInfo().SupportsFP16()) {
    return {};
  }
  return env_.GetSupportedStorages();
}

const GpuInfo& ClExecutionEnvironment::GetGpuInfo() const {
  return env_.GetDevicePtr()->GetInfo();
}

absl::Status ClExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  const OperationDef& op_def = operation->GetDefinition();
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i]->GetBHWDCShape();
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(src[i].CreateFromDescriptor(*src_cpu[i], &env_.context()));
    operation->SetSrc(&src[i], i);
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_cpu[i]->GetBHWDCShape();
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    TensorDescriptor descriptor_with_shape = op_def.dst_tensors[i];
    descriptor_with_shape.SetBHWDCShape(dst_shape);
    RETURN_IF_ERROR(
        CreateTensor(env_.context(), descriptor_with_shape, &dst[i]));

    operation->SetDst(&dst[i], i);
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  {
    CreationContext creation_context;
    creation_context.device = env_.GetDevicePtr();
    creation_context.context = &env_.context();
    creation_context.queue = env_.queue();
    creation_context.cache = env_.program_cache();
    RETURN_IF_ERROR(cl_op.Compile(creation_context));
  }
  RETURN_IF_ERROR(cl_op.UpdateParams());
  RETURN_IF_ERROR(cl_op.AddToQueue(env_.queue()));
  RETURN_IF_ERROR(env_.queue()->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    RETURN_IF_ERROR(dst[i].ToDescriptor(dst_cpu[i], env_.queue()));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
