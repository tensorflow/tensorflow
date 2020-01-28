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

#include "tensorflow/lite/delegates/gpu/metal/kernels/padding.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetPaddingCode(const PadAttributes& attr) {
  const std::string channels[] = {".x", ".y", ".z", ".w"};
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct uniforms {
      int4 src_size;
      int4 dst_size;
      int4 padding;
    };)";
  if (attr.type == PaddingContentType::REFLECT) {
    code += R"(
    int reflect(int x, int size) {
      return size - 1 - abs(abs(x) - size + 1);
    })";
  }
  code += R"(
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= params.dst_size.x ||
          static_cast<int>(gid.y) >= params.dst_size.y) {
        return;
      }

      FLT4 value = FLT4(0.0f);
      int s_x = static_cast<int>(gid.x) - params.padding.x;
      int s_y = static_cast<int>(gid.y) - params.padding.y;)";
  if (attr.type == PaddingContentType::REFLECT) {
    code += R"(
      s_x = reflect(s_x, params.src_size.x);
      s_y = reflect(s_y, params.src_size.y);
)";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      code +=
          "      int buffer_index = (int(gid.z) * params.src_size.y + s_y) * "
          "params.src_size.x + s_x;\n";
      code += "      value = src_buffer[buffer_index];\n";
    } else {
      code += "      int start_channel = static_cast<int>(gid.z) * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        code += "      {\n";
        code += "        int channel = start_channel + " + std::to_string(i) +
                ";\n";
        code += "        int s_z = channel - params.padding.z;\n";
        code += "        s_z = reflect(s_z, params.src_size.z);\n";
        code +=
            "        int buffer_index = ((s_z / 4) * params.src_size.y + s_y) "
            "* params.src_size.x + s_x;\n";
        code += "        FLT4 t = src_buffer[buffer_index];\n";
        code += "        FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        code += "        value" + s + " = t_ar[s_z % 4];\n";
        code += "      }\n";
      }
    }
  } else {
    code += R"(
      bool inside_x = s_x >= 0 && s_x < params.src_size.x;
      bool inside_y = s_y >= 0 && s_y < params.src_size.y;
      if (inside_x && inside_y) {
        int start_channel = static_cast<int>(gid.z) * 4;
    )";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      code +=
          "        int buffer_index = (int(gid.z) * params.src_size.y + s_y) * "
          "params.src_size.x + s_x;\n";
      code += "        value = src_buffer[buffer_index];\n";
    } else if (attr.prepended.c % 4 == 0) {
      code += R"(
        int s_z = static_cast<int>(gid.z) - params.padding.z / 4;
        if (s_z >= 0 && s_z < params.src_size.w) {
          int buffer_index = (s_z * params.src_size.y + s_y) * params.src_size.x + s_x;
          value = src_buffer[buffer_index];
        })";
    } else {
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        code += "    {\n";
        code +=
            "    int channel = start_channel + " + std::to_string(i) + ";\n";
        code += "    int s_z = channel - params.padding.z;\n";
        code += "    if (s_z >= 0 && s_z < params.src_size.z) {\n";
        code +=
            "      int buffer_index = ((s_z / 4) * params.src_size.y + s_y) * "
            "params.src_size.x + "
            "s_x;\n";
        code += "      FLT4 t = src_buffer[buffer_index];\n";
        code += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        code += "      value" + s + " = t_ar[s_z % 4];\n";
        code += "    }\n";
        code += "    }\n";
      }
    }
    code += "  }\n";
  }
  code +=
      "  int linear_index = (gid.z * params.dst_size.y + int(gid.y)) * "
      "params.dst_size.x + "
      "int(gid.x);\n";
  code += "  $2\n";
  code += "  dst_buffer[linear_index] = value;\n";
  code += "}\n";
  return code;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> Padding(int id, ValueId input_id,
                                              ValueId output_id,
                                              const PadAttributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetPaddingCode(attr);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, attr);
      }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             // int4 src_size
             dimension.w,
             dimension.h,
             dimension.c,
             IntegralDivideRoundUp(dimension.c, 4),
             // int4 dst_size
             output_dimension.w,
             output_dimension.h,
             output_dimension.c,
             IntegralDivideRoundUp(output_dimension.c, 4),
             // int4 prepended padding
             attr.prepended.w,
             attr.prepended.h,
             attr.prepended.c,
             attr.prepended.b,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [input_id,
                           attr](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{16, 16, 1};
    const auto& src_shape = buffers.find(input_id)->second;
    BHWC dst_shape = CalculateOutputShape(src_shape, attr);
    const int dst_layers = IntegralDivideRoundUp(dst_shape.c, 4);
    int groups_x = IntegralDivideRoundUp(dst_shape.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(dst_shape.h, groups_size.y);
    int groups_z = IntegralDivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
