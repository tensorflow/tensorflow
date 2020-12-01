/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/kernels/resize.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

std::string GetResizeBilinearCode(const Resize2DAttributes& attr) {
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.z || int(gid.y) >= size.w) {
        return;
      })";
  if (attr.half_pixel_centers) {
    code += "const float2 tex_coord = (float2(gid.xy) + 0.5f) * scale - 0.5f;";
  } else {
    code += "const float2 tex_coord = float2(gid.xy) * scale;";
  }
  code += R"(
      const float2 tex_coord_floor = floor(tex_coord);
      const int2 itex_coord_floor = int2(tex_coord_floor);
      const int2 borders = size.xy - int2(1, 1);
      int4 st;
      st.xy = max(itex_coord_floor, int2(0, 0));
      st.zw = min(itex_coord_floor + int2(1, 1), borders);
      const float2 t = tex_coord - tex_coord_floor; // interpolating factors
      const int src_index0 = (gid.z * size.y + st.y) * size.x + st.x;
      const int src_index1 = (gid.z * size.y + st.y) * size.x + st.z;
      const int src_index2 = (gid.z * size.y + st.w) * size.x + st.x;
      const int src_index3 = (gid.z * size.y + st.w) * size.x + st.z;
      FLT4 tex11 = src_buffer[src_index0];
      FLT4 tex21 = src_buffer[src_index1];
      FLT4 tex12 = src_buffer[src_index2];
      FLT4 tex22 = src_buffer[src_index3];
      // bilinear interpolation
      FLT4 value = mix(mix(tex11, tex21, static_cast<FLT>(t.x)),
                       mix(tex12, tex22, static_cast<FLT>(t.x)), static_cast<FLT>(t.y));
      const int linear_index = (gid.z * size.w + gid.y) * size.z + gid.x;
      $2
      output_buffer[linear_index] = value;
    }
  )";
  return code;
}

std::string GetResizeNearestCode(const Resize2DAttributes& attr) {
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.z || int(gid.y) >= size.w) {
        return;
      }
)";
  std::string fxc;
  std::string fyc;
  if (attr.half_pixel_centers) {
    fxc = "(float(gid.x) + 0.5f) * scale.x";
    fyc = "(float(gid.y) + 0.5f) * scale.y";
  } else {
    fxc = "float(gid.x) * scale.x";
    fyc = "float(gid.y) * scale.y";
  }
  if (attr.align_corners) {
    fxc += " + 0.5f";
    fyc += " + 0.5f";
  }
  code += "  int2 coord;\n";
  code += "  coord.x = static_cast<int>(" + fxc + ");\n";
  code += "  coord.y = static_cast<int>(" + fyc + ");\n";
  code += "  coord.x = max(0, coord.x);\n";
  code += "  coord.y = max(0, coord.y);\n";
  code += "  coord.x = min(coord.x, size.x - 1);\n";
  code += "  coord.y = min(coord.y, size.y - 1);\n";
  code += R"(
      const int src_index = (gid.z * size.y + coord.y) * size.x + coord.x;
      FLT4 value = src_buffer[src_index];
      const int linear_index = (gid.z * size.w + gid.y) * size.z + gid.x;
      $2
      output_buffer[linear_index] = value;
    }
  )";
  return code;
}

ComputeTaskDescriptor Resize(ValueId input_id, ValueId output_id,
                             const Resize2DAttributes& attr) {
  ComputeTaskDescriptor desc;
  switch (attr.type) {
    case SamplingType::BILINEAR:
      desc.shader_source = GetResizeBilinearCode(attr);
      break;
    case SamplingType::NEAREST:
      desc.shader_source = GetResizeNearestCode(attr);
      break;
    default:
      // Unknown sampling type
      return {};
  }

  desc.input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc.output_buffer = {output_id, "device FLT4* output_buffer"};

  desc.uniform_buffers = {
      {"constant int4& size",
       [](const std::vector<BHWC>& src_shapes,
          const std::vector<BHWC>& dst_shapes) {
         std::vector<int> sizes = {
             src_shapes[0].w,
             src_shapes[0].h,
             dst_shapes[0].w,
             dst_shapes[0].h,
         };
         return GetByteBuffer(sizes);
       }},
      {"constant float2& scale",
       [attr](const std::vector<BHWC>& src_shapes,
              const std::vector<BHWC>& dst_shapes) {
         std::vector<float> sizes = {
             CalculateResizeScale(src_shapes[0].w, dst_shapes[0].w, attr),
             CalculateResizeScale(src_shapes[0].h, dst_shapes[0].h, attr),
         };
         return GetByteBuffer(sizes);
       }},
  };

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{16, 16, 1};
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    const int dst_layers = DivideRoundUp(dst_shapes[0].c, 4);
    int groups_z = DivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
