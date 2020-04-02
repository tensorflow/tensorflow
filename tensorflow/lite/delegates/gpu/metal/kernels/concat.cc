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

#include "tensorflow/lite/delegates/gpu/metal/kernels/concat.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetConcatZCode(const std::vector<int> channels) {
  const std::string postfix[] = {".x", ".y", ".z", ".w"};
  const std::string postfix_2[] = {".x", ".xy", ".xyz", ""};
  const std::string types[] = {"FLT", "FLT2", "FLT3", "FLT4"};
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    struct uniforms {
      int4 src_size;
    };

    $0
    kernel void ComputeFunction(
                                $1
                                uint2 ugid[[thread_position_in_grid]]) {
    if (static_cast<int>(ugid.x) >= params.src_size.x ||
        static_cast<int>(ugid.y) >= params.src_size.y) {
      return;
    }

    FLT4 value = FLT4(0.0f);
    const int xy_offset = int(ugid.y) * params.src_size.x + int(ugid.x);
    int linear_index = xy_offset;
  )";

  int out_channel = 0;
  int read_index = 0;
  int dst_z = 0;
  for (int i = 0; i < channels.size(); ++i) {
    const int depth = IntegralDivideRoundUp(channels[i], 4);
    code += "  {\n";
    code += "  int src_address = xy_offset;\n";
    for (int d = 0; d < depth; ++d) {
      const int channels_in_group = std::min(4, channels[i] - d * 4);
      const std::string temp_name = "t" + std::to_string(read_index);
      code += "  " + types[channels_in_group - 1] + " " + temp_name + " = " +
              "src_buffer" + std::to_string(i) + "[src_address]" +
              postfix_2[channels_in_group - 1] + ";\n";
      code += "  src_address += params.src_size.w;\n";
      for (int c = 0; c < channels_in_group; ++c) {
        if (channels_in_group == 1) {
          code += "  value" + postfix[out_channel] + " = " + temp_name + ";\n";
        } else {
          code += "  value" + postfix[out_channel] + " = " + temp_name +
                  postfix[c] += ";\n";
        }
        out_channel++;
        if (out_channel == 4) {
          out_channel = 0;
          code += "  {\n";
          code += "    uint3 gid = uint3(ugid.x, ugid.y, " +
                  std::to_string(dst_z) + ");\n";
          code += "    $2\n";
          code += "    dst_buffer[linear_index] = value;\n";
          code += "    linear_index += params.src_size.w;\n";
          code += "  }\n";
          dst_z++;
        }
      }
      read_index++;
    }
    code += "  }\n";
  }
  if (out_channel != 0) {
    code += "  {\n";
    code += "    uint3 gid = uint3(ugid.x, ugid.y, " + std::to_string(dst_z) +
            ");\n";
    code += "    $2\n";
    code += "    dst_buffer[linear_index] = value;\n";
    code += "  }\n";
  }
  code += "}\n";
  return code;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> ConcatZ(
    int id, std::vector<ValueId> input_ids, ValueId output_id,
    const ConcatAttributes& attr, const std::vector<BHWC>& input_shapes) {
  std::vector<int> channels;
  channels.reserve(input_shapes.size());
  for (const auto& shape : input_shapes) {
    channels.push_back(shape.c);
  }
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetConcatZCode(channels);

  for (int i = 0; i < input_ids.size(); ++i) {
    const std::string buffer_name =
        "device FLT4* const src_buffer" + std::to_string(i);
    desc->input_buffers.push_back({input_ids[i], buffer_name});
  }

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_ids, attr](const std::map<ValueId, BHWC>& buffers) {
        std::vector<BHWC> src_shapes(input_ids.size());
        for (int i = 0; i < input_ids.size(); ++i) {
          src_shapes[i] = buffers.find(input_ids[i])->second;
        }
        BHWC dst_shape;
        CalculateOutputShape(src_shapes, attr, &dst_shape).IgnoreError();
        return dst_shape;
      }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_ids](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_ids[0])->second;
         std::vector<int> uniform_params{
             dimension.w,
             dimension.h,
             0,
             dimension.w * dimension.h,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [input_ids](const std::map<ValueId, BHWC>& buffers) {
    const auto& src_dim = buffers.find(input_ids[0])->second;
    const uint3 groups_size{16, 16, 1};
    int groups_x = IntegralDivideRoundUp(src_dim.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(src_dim.h, groups_size.y);
    int groups_z = 1;
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> ConcatX(
    int id, std::vector<ValueId> input_ids, ValueId output_id,
    const ConcatAttributes& attr, const std::vector<BHWC>& input_shapes) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;

  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
        return;
      }
      FLT4 value;
    )";
  int output_width = 0;
  for (int buffer_index = 0; buffer_index < input_shapes.size();
       buffer_index++) {
    const auto& dims = input_shapes[buffer_index];
    output_width += dims.w;

    // Generated shader example:
    // if (gid.x < 10) value = src_buffer0[(gid.y + gid.z * 3) * 4 + gid.x - 3];
    // else
    if (buffer_index < input_shapes.size() - 1) {
      code += "if (gid.x < " + std::to_string(output_width) + ")";
    }
    code += "value = src_buffer" + std::to_string(buffer_index) +
            "[(gid.y + gid.z * " + std::to_string(dims.h) + ") * " +
            std::to_string(dims.w) + " + gid.x - " +
            std::to_string(output_width - dims.w) + "];\n";
    if (buffer_index < input_shapes.size() - 1) {
      code += "else ";
    }
  }
  code += "const int linear_index = (gid.y + gid.z * " +
          std::to_string(input_shapes[0].h) + ") * " +
          std::to_string(output_width) + " + gid.x;";
  code += R"(
      $2
      dst_buffer[linear_index] = value;
    }
  )";
  desc->shader_source = code;

  for (int i = 0; i < input_ids.size(); ++i) {
    const std::string buffer_name =
        "device FLT4* const src_buffer" + std::to_string(i);
    desc->input_buffers.push_back({input_ids[i], buffer_name});
  }

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_ids, attr](const std::map<ValueId, BHWC>& buffers) {
        std::vector<BHWC> src_shapes(input_ids.size());
        for (int i = 0; i < input_ids.size(); ++i) {
          src_shapes[i] = buffers.find(input_ids[i])->second;
        }
        BHWC dst_shape;
        CalculateOutputShape(src_shapes, attr, &dst_shape).IgnoreError();
        return dst_shape;
      }};

  desc->uniform_buffers = {
      {"constant int3& size",
       [output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{dimension.w, dimension.h,
                                         IntegralDivideRoundUp(dimension.c, 4),
                                         /*padding=*/0};
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const uint3 groups_size{1, 1, 1};
    int groups_x = IntegralDivideRoundUp(output_dims.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(output_dims.h, groups_size.y);
    int groups_z = IntegralDivideRoundUp(output_dims.c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> ConcatY(
    int id, std::vector<ValueId> input_ids, ValueId output_id,
    const ConcatAttributes& attr, const std::vector<BHWC>& input_shapes) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;

  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
        return;
      }
      FLT4 value;
  )";
  int output_height = 0;
  for (int buffer_index = 0; buffer_index < input_shapes.size();
       buffer_index++) {
    const auto& dims = input_shapes[buffer_index];
    output_height += dims.h;

    // Generated shader example:
    // if (gid.y < 10) value = src_buffer0[(gid.y - 3 + gid.z * 5) * 4 + gid.x];
    // else
    if (buffer_index < input_shapes.size() - 1) {
      code += "if (gid.y < " + std::to_string(output_height) + ")";
    }
    code += "value = src_buffer" + std::to_string(buffer_index) + "[(gid.y - " +
            std::to_string(output_height - dims.h) + " + gid.z * " +
            std::to_string(dims.h) + ") * " + std::to_string(dims.w) +
            " + gid.x];\n";
    if (buffer_index < input_shapes.size() - 1) {
      code += "else ";
    }
  }
  const auto& dims = input_shapes[0];
  code += "const int linear_index = (gid.y + gid.z * " +
          std::to_string(output_height) + ") * " + std::to_string(dims.w) +
          " + gid.x;";
  code += R"(
      $2
      dst_buffer[linear_index] = value;
    }
  )";
  desc->shader_source = code;

  for (int i = 0; i < input_ids.size(); ++i) {
    const std::string buffer_name =
        "device FLT4* const src_buffer" + std::to_string(i);
    desc->input_buffers.push_back({input_ids[i], buffer_name});
  }

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_ids, attr](const std::map<ValueId, BHWC>& buffers) {
        std::vector<BHWC> src_shapes(input_ids.size());
        for (int i = 0; i < input_ids.size(); ++i) {
          src_shapes[i] = buffers.find(input_ids[i])->second;
        }
        BHWC dst_shape;
        CalculateOutputShape(src_shapes, attr, &dst_shape).IgnoreError();
        return dst_shape;
      }};

  desc->uniform_buffers = {
      {"constant int3& size",
       [output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{dimension.w, dimension.h,
                                         IntegralDivideRoundUp(dimension.c, 4),
                                         /*padding=*/0};
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const uint3 groups_size{1, 1, 1};
    int groups_x = IntegralDivideRoundUp(output_dims.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(output_dims.h, groups_size.y);
    int groups_z = IntegralDivideRoundUp(output_dims.c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> Concat(
    int id, std::vector<ValueId> input_ids, ValueId output_id,
    const ConcatAttributes& attr, const std::vector<BHWC>& input_shapes) {
  if (attr.axis == Axis::CHANNELS) {
    return ConcatZ(id, input_ids, output_id, attr, input_shapes);
  } else if (attr.axis == Axis::WIDTH) {
    return ConcatX(id, input_ids, output_id, attr, input_shapes);
  } else {
    return ConcatY(id, input_ids, output_id, attr, input_shapes);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
