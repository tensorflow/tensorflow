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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_DESCRIPTOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_DESCRIPTOR_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace metal {

using UpdateArgsFunction = std::function<absl::Status(
    const std::vector<BHWC>& src_shapes, const std::vector<BHWC>& dst_shapes,
    ArgumentsBinder* args)>;
using DispatchParamsFunction = std::function<std::pair<uint3, uint3>(
    const std::vector<BHWC>& src_shapes, const std::vector<BHWC>& dst_shapes)>;

// Compute task descriptor contains a linkable shader code or a code for
// complete shader to which other linkable can be attached or not. An operation
// can produce one or more descriptors and graph compiler uses descriptors as
// building blocks. All required data like immutable operation parameters
// (weights etc.) is attached to the descriptor.
struct ComputeTaskDescriptor {
  ComputeTaskDescriptor() = default;
  explicit ComputeTaskDescriptor(const OperationDef& def);
  // Move only
  ComputeTaskDescriptor(ComputeTaskDescriptor&& task) = default;
  ComputeTaskDescriptor& operator=(ComputeTaskDescriptor&& task) = default;
  ComputeTaskDescriptor(const ComputeTaskDescriptor&) = delete;
  ComputeTaskDescriptor& operator=(const ComputeTaskDescriptor&) = delete;

  OperationDef definition;
  Arguments args;
  bool is_linkable = false;
  // A linkable function or a full shader source with 3 parameters $ for
  // substitute function. Example of linkable: "(FLT4 linkable$0(FLT4 value, int
  // linear_index) { return value; })" Example of non-linkable function:
  // #include <metal_stdlib>
  // using namespace metal;
  // $0
  // kernel void ComputeFunction(
  //                             $1
  //                             uint3 gid[[thread_position_in_grid]]) {
  //   if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
  //     return;
  //   }
  //   const int linear_index = (gid.z * size.y + gid.y) * size.x + gid.x;
  //   FLT4 value = input_buffer[linear_index] + 1.0f;
  //   $2
  //   output_buffer[linear_index] = value;
  // }

  std::string shader_source;
  std::vector<std::string> src_tensors_names;
  std::vector<std::string> dst_tensors_names;
  UpdateArgsFunction update_function = {
      [](const std::vector<BHWC>& src_shapes,
         const std::vector<BHWC>& dst_shapes,
         ArgumentsBinder* args) -> absl::Status { return absl::OkStatus(); }};
  // Dynamic resizing of input tensor is supported. User-defined functions to
  // calculate new parameters for GPU compute task dispatching. A leading
  // unlinkable task must provide this.
  DispatchParamsFunction resize_function;

  void AddSrcTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddDstTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);

  absl::Status AddTask(ComputeTaskDescriptor* task_desc);
  absl::Status AddOperation(GPUOperation* operation);
  void AssembleCode();

 private:
  friend class ComputeTask;
  int linkable_count = 0;        // temporary, used during op construction
  std::string elementwise_code;  // temporary, used during op construction
};

/// Helper function to convert buffer's content into stream of bytes
template <typename T>
std::vector<uint8_t> GetByteBuffer(const std::vector<T>& input_vector) {
  std::vector<uint8_t> result;
  result.insert(result.begin(),
                reinterpret_cast<const uint8_t*>(input_vector.data()),
                reinterpret_cast<const uint8_t*>(input_vector.data()) +
                    input_vector.size() * sizeof(*input_vector.data()));
  return result;
}

/// Converts float to destination type (if needed) and stores as bytes array.
/// supports DataType::FLOAT32 and DataType::FLOAT16
std::vector<uint8_t> GetByteBufferConverted(
    const std::vector<float>& input_vector, DataType data_type);

/// Resizes, Converts float to destination type (if needed) and stores as bytes
/// array.
/// supports DataType::FLOAT32 and DataType::FLOAT16
std::vector<uint8_t> GetByteBufferConvertedResized(
    const std::vector<float>& input_vector, DataType data_type,
    size_t elements_count);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_DESCRIPTOR_H_
