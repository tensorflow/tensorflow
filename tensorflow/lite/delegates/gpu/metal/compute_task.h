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

  /// Returns empty string or error if shader can't be compiled.
  absl::Status CompileWithDevice(id<MTLDevice> device,
                                 const NodeDescriptor& desc,
                                 CalculationsPrecision precision);

  /// Updates parameters for inputs/outputs/intermediate tensors
  absl::Status UpdateParamsWithDevice(
      id<MTLDevice> device, const std::map<ValueId, BHWC>& tensor_shapes);

  bool HasInOutIds(const std::set<ValueId>& ids) const;

  void EncodeWithEncoder(id<MTLComputeCommandEncoder> encoder);

  std::vector<ValueId> GetOutputIds() const;
  std::vector<ValueId> GetInputIds() const;

  void SetSrcTensor(const MetalSpatialTensor& tensor, int index);

  void SetDstTensor(const MetalSpatialTensor& tensor, int index);

  void SetDescription(const std::string& description);

 private:
  struct InputBuffer {
    ValueId uid;
    id<MTLBuffer> metal_handle;
  };

  struct OutputBuffer {
    ValueId uid;
    id<MTLBuffer> metal_handle;
  };

  struct UniformBuffer {
    std::vector<uint8_t> data;
    UniformsFunction data_function;
  };

  id<MTLComputePipelineState> program_;
  std::vector<InputBuffer> input_buffers_;
  std::vector<OutputBuffer> output_buffers_;
  std::vector<id<MTLBuffer>> immutable_buffers_;
  std::vector<UniformBuffer> uniform_buffers_;
  uint3 groups_size_;
  uint3 groups_count_;
  DispatchParamsFunction resize_function_;
  std::string description_;
  MetalArguments metal_args_;
  std::vector<std::string> src_tensors_names_;
  std::vector<std::string> dst_tensors_names_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
