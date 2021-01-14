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

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_arguments.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

class ComputeTask {
 public:
  ComputeTask() = default;

  // Move only
  ComputeTask(ComputeTask&& args) = default;
  ComputeTask& operator=(ComputeTask&& args) = default;
  ComputeTask(const ComputeTask&) = delete;
  ComputeTask& operator=(const ComputeTask&) = delete;

  void Init(std::unique_ptr<ComputeTaskDescriptor>&& task_desc);

  ComputeTaskDescriptor& GetTaskDesc() { return *task_desc_; }
  const ComputeTaskDescriptor& GetTaskDesc() const { return *task_desc_; }

  /// Returns empty string or error if shader can't be compiled.
  absl::Status Compile(CalculationsPrecision precision, MetalDevice* device);

  /// Updates parameters for inputs/outputs/intermediate tensors
  absl::Status UpdateParams(const GpuInfo& gpu_info,
                            const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes);

  void EncodeWithEncoder(id<MTLComputeCommandEncoder> encoder);

  void SetSrcTensor(const MetalSpatialTensor& tensor, int index);

  void SetDstTensor(const MetalSpatialTensor& tensor, int index);

 private:
  std::unique_ptr<ComputeTaskDescriptor> task_desc_;
  id<MTLComputePipelineState> program_;
  MetalArguments metal_args_;
  uint3 groups_size_;
  uint3 groups_count_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
