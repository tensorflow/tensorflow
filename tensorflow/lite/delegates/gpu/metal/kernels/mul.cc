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

#include "tensorflow/lite/delegates/gpu/metal/kernels/mul.h"

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
namespace {

std::string GetMaxUnpoolingCode() {
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    struct uniforms {
      int4 src_size;
      int4 dst_size;
    };

    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      int X = static_cast<int>(gid.x);
      int Y = static_cast<int>(gid.y);
      if (X >= params.dst_size.x || Y >= params.dst_size.y) {
        return;
      }
      int src_0_index = (gid.z * params.src_size.y + static_cast<int>(gid.y)) *
                        params.src_size.x + static_cast<int>(gid.x);
      int src_1_index = 0;
      if (params.dst_size.z == 1) {
        // [H, W, C] x [H, W, 0][0]
        src_1_index = static_cast<int>(gid.y) * params.src_size.x +
                      static_cast<int>(gid.x);
      } else if (params.src_0_size.y == params.src_1_size.y &&
                 params.src_0_size.x == params.src_1_size.x) {
        // [H, W, C] x [H, W, C]
        src_1_index = src_0_index;
      } else {
        // [H, W, C] x [0, 0, C]
        src_1_index = gid.z * params.src_size.y * params.src_size.x ;
      }
      FLT4 value = src_buffer_0[src_index] * src_buffer_1[src_1_index];
      $2
      output_buffer[linear_index] = value;
    }
  )";
  return shader_source;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> ApplyMask(int id, ValueId input_id_0,
                                                ValueId input_id_1,
                                                ValueId output_id,
                                                const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetMaxUnpoolingCode();

  desc->input_buffers = {
      {input_id_0, "device FLT4* const src_buffer_0"},  // data
      {input_id_1, "device FLT4* const src_buffer_1"},  // mask
  };

  desc->output_buffer = {
      output_id, "device FLT4* output_buffer",
      [input_id_0, input_id_1](const std::map<ValueId, BHWC>& buffers) {
        return buffers.find(input_id_0)->second;
      }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id_0, input_id_1,
        output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dim_0 = buffers.find(input_id_0)->second;
         const auto& input_dim_1 = buffers.find(input_id_1)->second;
         const auto& output_dim = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             input_dim_0.w, input_dim_0.h, input_dim_0.c, 0,
             input_dim_1.w, input_dim_1.h, input_dim_1.c, 0,
             output_dim.w,  output_dim.h,  output_dim.c,  0,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [input_id_0,
                           input_id_1](const std::map<ValueId, BHWC>& buffers) {
    const auto& src_shape = buffers.find(input_id_0)->second;
    const uint3 groups_size{16, 16, 1};
    int groups_x = IntegralDivideRoundUp(src_shape.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(src_shape.h, groups_size.y);
    int groups_z = IntegralDivideRoundUp(src_shape.c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> Multiply(
    int id, ValueId input_id, ValueId output_id,
    const MultiplyScalarAttributes& attr, const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  auto multiplier = absl::get_if<float>(&attr.param);
  auto mul_buffer =
      absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
  const bool scalar = multiplier != nullptr;
  const std::string param_desc =
      scalar ? "float multiplier" : "device FLT4* const mul_buf";
  std::string code =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, ";
  code += param_desc + ") {\n";
  if (scalar) {
    code += "return value * multiplier;\n";
  } else {
    code += "return value * mul_buf[gid.z];\n";
  }
  code += "}\n";
  desc->shader_source = code;
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  if (scalar) {
    std::vector<uint8_t> multiplier_bits =
        GetByteBuffer(std::vector<float>{*multiplier});
    desc->uniform_buffers = {
        {"constant float&",
         [multiplier_bits](const std::map<ValueId, BHWC>& buffers) {
           return multiplier_bits;
         }},
    };
  } else {
    desc->immutable_buffers = {
        {"device FLT4* const",
         GetByteBufferConverted(mul_buffer->data, options.storage_precision)},
    };
  }
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
