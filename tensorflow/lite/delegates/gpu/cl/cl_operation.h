/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_OPERATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_OPERATION_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {
namespace cl {

struct CreationContext {
  const CLDevice* device;
  CLContext* context;
  CLCommandQueue* queue;
  ProgramCache* cache;

  const GpuInfo& GetGpuInfo() const { return device->info_; }
};

class ClOperation {
 public:
  ClOperation() = default;
  virtual ~ClOperation() = default;
  // Move only
  ClOperation(ClOperation&& operation) = default;
  ClOperation& operator=(ClOperation&& operation) = default;
  ClOperation(const ClOperation&) = delete;
  ClOperation& operator=(const ClOperation&) = delete;

  void Init(std::unique_ptr<GPUOperation>&& gpu_operation) {
    operation_ = std::move(gpu_operation);
  }

  GPUOperation& GetGpuOperation() { return *operation_; }
  const GPUOperation& GetGpuOperation() const { return *operation_; }
  uint64_t GetKernelFingerprint() const { return kernel_fingerprint_; }

  const OperationDef& GetDefinition() const {
    return operation_->GetDefinition();
  }

  absl::Status AddOperation(ClOperation* operation);

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  absl::Status SetSrcTensor(int index, Tensor* tensor);
  absl::Status SetDstTensor(int index, Tensor* tensor);

  absl::Status AddToQueue(CLCommandQueue* queue) {
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    return queue->Dispatch(kernel_, operation_->GetWorkGroupsCount(),
                           operation_->work_group_size_);
  }

  // for better profiling
  absl::Status AddToQueueNTimes(ProfilingCommandQueue* queue, int n,
                                int flush_period = 0) {
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    return queue->DispatchNTimes(kernel_, operation_->GetWorkGroupsCount(),
                                 operation_->work_group_size_, n, flush_period);
  }

  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);

  absl::Status Compile(const CreationContext& creation_context);

  absl::Status RestoreDeserialized(const ProgramCache& program_cache,
                                   uint64_t fingerprint,
                                   const GpuInfo& gpu_info,
                                   const int3& work_group_size,
                                   CLContext* context);

  int3 GetWorkGroupSize() const { return operation_->work_group_size_; }

 private:
  std::unique_ptr<GPUOperation> operation_;
  CLKernel kernel_;
  uint64_t kernel_fingerprint_;
  CLArguments cl_args_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_OPERATION_H_
