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

#include "tensorflow/lite/delegates/gpu/metal/kernels/slice.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
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

std::string GetSliceCode(const SliceAttributes& attr) {
  std::stringstream code;

  code << R"(
    #include <metal_stdlib>
    using namespace metal;

    struct uniforms {
      int4 src_size;
      int4 dst_size;
    };

    constant int4 width = int4($0, $1, $2, 0);
    constant int4 height = int4($3, $4, $5, 0);
    constant int4 channels = int4($6, $7, $8, 0);
    constant FLT4 null_vec = FLT4(0.0f, 0.0f, 0.0f, 0.0f);

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= params.dst_size.x ||
          static_cast<int>(gid.y) >= params.dst_size.y) {
        return;
      }

      FLT4 value;
      short2 offset;
  )";
  if (attr.strides.w > 0) {
    code << "      offset.x = width.x;" << std::endl;
  } else {
    if (attr.ends.w > 0) {
      code << "      offset.x = width.z;" << std::endl;
    } else {
      code << "      offset.x = params.src_size.x + width.z;" << std::endl;
    }
  }
  if (attr.strides.h > 0) {
    code << "      offset.y = height.x;" << std::endl;
  } else {
    if (attr.ends.h > 0) {
      code << "      offset.y = height.z;" << std::endl;
    } else {
      code << "      offset.y = params.src_size.y + height.z;" << std::endl;
    }
  }
  code << std::endl;
  code << "      short2 stride = short2(width.y, height.y);" << std::endl;

  code << "      const short2 s_c = offset + short2(gid.xy) * stride;"
       << std::endl;
  code << "      bool outside = false;" << std::endl;
  code << "      int step = gid.z * 4;" << std::endl;
  code << "      FLT4 tmp;" << std::endl;
  code << "      int buffer_index = 0;" << std::endl;
  code << "      int addr = 0;" << std::endl;
  code << std::endl;
  for (int i = 0; i < 4; i++) {
    code << "      addr = step * channels.y;" << std::endl;
    if (attr.strides.c > 0) {
      code << "      addr += channels.x;" << std::endl;
    } else {
      if (attr.ends.c > 0) {
        code << "      addr += channels.z;" << std::endl;
      } else {
        code << "      addr += params.src_size.z + channels.z;" << std::endl;
      }
    }
    code << "      buffer_index = ((addr / 4) * params.src_size.y + s_c.y) * "
            "params.src_size.x + "
            "s_c.x;"
         << std::endl;
    code << "      outside = step >= params.dst_size.z;" << std::endl;
    code << "      tmp = outside ? null_vec : src_buffer[buffer_index];"
         << std::endl;
    code << "      value[" << i << "] = tmp[addr % 4];" << std::endl;
    if (i != 3) {
      code << "      step++;" << std::endl;
      code << std::endl;
    }
  }
  code << R"(
      int linear_index = (gid.z * params.dst_size.y + int(gid.y)) *
        params.dst_size.x + int(gid.x);
      $$2
      dst_buffer[linear_index] = value;
    })";
  return absl::Substitute(
      code.str(), attr.starts.w, attr.strides.w, attr.ends.w, attr.starts.h,
      attr.strides.h, attr.ends.h, attr.starts.c, attr.strides.c, attr.ends.c);
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> Slice(int id, ValueId input_id,
                                            ValueId output_id,
                                            const SliceAttributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetSliceCode(attr);

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
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             // int4 src_size
             dimension.w,
             dimension.h,
             dimension.c,
             DivideRoundUp(dimension.c, 4),
             // int4 dst_size
             output_dimension.w,
             output_dimension.h,
             output_dimension.c,
             DivideRoundUp(output_dimension.c, 4),
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [input_id,
                           attr](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{16, 16, 1};
    const auto& src_shape = buffers.find(input_id)->second;
    BHWC dst_shape = CalculateOutputShape(src_shape, attr);
    int groups_x = DivideRoundUp(dst_shape.w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shape.h, groups_size.y);
    const int dst_layers = DivideRoundUp(dst_shape.c, 4);
    int groups_z = DivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
