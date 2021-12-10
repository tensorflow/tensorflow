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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_

#import <Metal/Metal.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tuning_type.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_arguments.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

class ComputeTask {
 public:
  ComputeTask() = default;
  ~ComputeTask();

  // Move only
  ComputeTask(ComputeTask&& task);
  ComputeTask& operator=(ComputeTask&& task);
  ComputeTask(const ComputeTask&) = delete;
  ComputeTask& operator=(const ComputeTask&) = delete;

  void Init(std::unique_ptr<GPUOperation>&& operation);

  const OperationDef& GetDefinition() const;
  const GPUOperation& GetGpuOperation() const { return *operation_; }
  bool IsLinkable() const;

  absl::Status AddTask(ComputeTask* task);

  absl::Status Compile(MetalDevice* device);

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  void Encode(id<MTLComputeCommandEncoder> encoder);

  API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
  void EncodeToICB(id<MTLIndirectComputeCommand> icb_command);
  API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
  void AddResourcesToEncoder(id<MTLComputeCommandEncoder> encoder) const;

  void Update();

  void SetSrcTensor(MetalSpatialTensor* tensor, int index);

  void SetDstTensor(MetalSpatialTensor* tensor, int index);

  absl::Status Tune(TuningType tuning_type, MetalDevice* device);

 private:
  absl::Status CompileProgram(MetalDevice* device,
                              CalculationsPrecision precision,
                              const std::string& kernel_code);
  void Release();

  std::unique_ptr<GPUOperation> operation_;
  id<MTLComputePipelineState> program_ = nullptr;
  MetalArguments metal_args_;

  bool use_arguments_buffer_ = false;  // optional
  bool need_icb_support_ = false;      // optional
  id<MTLArgumentEncoder> arguments_encoder_ = nullptr;
  id<MTLBuffer> arg_buffer_ = nullptr;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
