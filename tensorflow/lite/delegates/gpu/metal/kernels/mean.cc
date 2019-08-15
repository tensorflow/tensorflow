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

#include "tensorflow/lite/delegates/gpu/metal/kernels/mean.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> Mean(
    int id, ValueId input_id, ValueId output_id,
    const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]],
                                uint3 tid[[thread_position_in_threadgroup]]
                                ) {
        if (int(gid.x) >= size.x || int(gid.y) >= size.y || int(tid.x) != 0 || int(tid.y) != 0) {
            return;
        }
        FLT4 sum = FLT4(0.0);
        int linear_index = 0;
        for (int x = 0; x < size.x; x++) {
          for (int y = 0; y < size.y; y++) {
            linear_index = (gid.z * size.y + y) * size.x + x;
            sum += input_buffer[linear_index];
          }
        }

        output_buffer[gid.z] = sum / (size.x * size.y);
    })";
  desc->shader_source = code;
  desc->input_buffers = {{input_id, "device FLT4* const input_buffer"}};
  desc->output_buffer = {output_id, "device FLT4* output_buffer", [input_id](const std::map<ValueId, BHWC>& buffers) {
    auto in_size = buffers.find(input_id)->second;
    return BHWC(in_size.b, 1, 1, in_size.c);
  }};
  desc->uniform_buffers = {
      {"constant int2& size", [input_id](const std::map<ValueId, BHWC>& buffers) {
          const auto& dimension = buffers.find(input_id)->second;
          std::vector<int> uniform_params = {dimension.w, dimension.h};
          return VectorToUint8Vector(uniform_params);
      }}
  };
  desc->resize_function = [input_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& src_dim = buffers.find(input_id)->second;
    const uint3 groups_size{src_dim.w, src_dim.h, 1};
    int groups_x = 1;
    int groups_y = 1;
    int groups_z = AlignByN(src_dim.c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
