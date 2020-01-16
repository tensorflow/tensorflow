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
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

using OutputDimensions =
    std::function<BHWC(const std::map<ValueId, BHWC>& buffers)>;
using UniformsFunction =
    std::function<std::vector<uint8_t>(const std::map<ValueId, BHWC>& buffers)>;
using DispatchParamsFunction = std::function<std::pair<uint3, uint3>(
    const std::map<ValueId, BHWC>& buffers)>;

// Compute task descriptor contains a linkable shader code or a code for
// complete shader to which other linkable can be attached or not. An operation
// can produce one or more descriptors and graph compiler uses descriptors as
// building blocks. All required data like immutable operation parameters
// (weights etc.) is attached to the descriptor.
struct ComputeTaskDescriptor {
  struct InputBufferDescriptor {
    ValueId id;
    // The declaration is inserted into the compute function arguments list.
    // Example for non-linkable task: "device FLT4* const input_buffer"
    // Example for linkable: "device FLT4* const"
    std::string declaration;
  };
  struct OutputBufferDescriptor {
    ValueId id;
    // The declaration is inserted into the compute function arguments list.
    // Example for non-linkable task: "device FLT4* output_buffer"
    // Example for linkable: "device FLT4*"
    std::string declaration;
    // Multiple outputs are allowed from a linkable operation so after fusion
    // each buffer's dimensions are calculated separately from different
    // operations.
    OutputDimensions dimensions_function;
    // Fusion absorbs intermediate tensors. Keep this ids to properly store
    // output dimensions.
    std::vector<ValueId> alias;
  };
  struct ImmutableBufferDescriptor {
    std::string declaration;
    std::vector<uint8_t> data;
  };
  // Uniforms are recalculated at any setInputDimensions call.
  struct UniformBufferDescriptor {
    // The declaration is inserted into the compute function arguments list.
    // Example: "constant uint4& some_uniforms"
    std::string declaration;
    // This function re-calculates uniforms for specific input dimensions.
    UniformsFunction data_function;
  };

  // Unique ID to match the graph compilation errors.
  int id;
  bool is_linkable;
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
  std::vector<InputBufferDescriptor> input_buffers;
  // A single per-operation output is supported now.
  OutputBufferDescriptor output_buffer;
  std::vector<ImmutableBufferDescriptor> immutable_buffers;
  std::vector<UniformBufferDescriptor> uniform_buffers;
  // Dynamic resizing of input tensor is supported. User-defined functions to
  // calculate new parameters for GPU compute task dispatching. A leading
  // unlinkable task must provide this.
  DispatchParamsFunction resize_function;
};

using ComputeTaskDescriptorPtr = std::shared_ptr<ComputeTaskDescriptor>;

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
std::vector<uint8_t> GetByteBufferConverted(
    const std::vector<float>& input_vector,
    RuntimeOptions::Precision destination_type);

/// Resizes, Converts float to destination type (if needed) and stores as bytes
/// array.
std::vector<uint8_t> GetByteBufferConvertedResized(
    const std::vector<float>& input_vector,
    RuntimeOptions::Precision destination_type, size_t elements_count);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_DESCRIPTOR_H_
