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
  std::string c = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  if (int(gid.x) >= args.dst_tensor.Width() || int(gid.y) >= args.dst_tensor.Height()) {
    return;
  }
)";
  if (attr.half_pixel_centers) {
    c += "  float2 tex_coord = (float2(gid.xy) + 0.5f) * scale - 0.5f;";
  } else {
    c += "  float2 tex_coord = float2(gid.xy) * scale;";
  }
  c += R"(
  float2 tex_coord_floor = floor(tex_coord);
  int2 itex_coord_floor = int2(tex_coord_floor);
  int2 borders = int2(args.src_tensor.Width() - 1, args.src_tensor.Height() - 1);
  int4 st;
  st.xy = max(itex_coord_floor, int2(0, 0));
  st.zw = min(itex_coord_floor + int2(1, 1), borders);
  float2 t = tex_coord - tex_coord_floor; // interpolating factors
  FLT4 tex11 = args.src_tensor.Read(st.x, st.y, gid.z);
  FLT4 tex21 = args.src_tensor.Read(st.z, st.y, gid.z);
  FLT4 tex12 = args.src_tensor.Read(st.x, st.w, gid.z);
  FLT4 tex22 = args.src_tensor.Read(st.z, st.w, gid.z);
  // bilinear interpolation
  FLT4 value = mix(mix(tex11, tex21, static_cast<FLT>(t.x)),
                   mix(tex12, tex22, static_cast<FLT>(t.x)), static_cast<FLT>(t.y));
  $2
  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);
}
)";
  return c;
}

std::string GetResizeNearestCode(const Resize2DAttributes& attr) {
  std::string c = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  if (int(gid.x) >= args.dst_tensor.Width() || int(gid.y) >= args.dst_tensor.Height()) {
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
  c += "  int2 coord;\n";
  c += "  coord.x = static_cast<int>(" + fxc + ");\n";
  c += "  coord.y = static_cast<int>(" + fyc + ");\n";
  c += "  coord.x = max(0, coord.x);\n";
  c += "  coord.y = max(0, coord.y);\n";
  c += "  coord.x = min(coord.x, args.src_tensor.Width() - 1);\n";
  c += "  coord.y = min(coord.y, args.src_tensor.Height() - 1);\n";
  c += R"(
  FLT4 value = args.src_tensor.Read(coord.x, coord.y, gid.z);
  args.dst_tensor.GetAddress(linear_index, gid.x, gid.y, gid.z);
  $2
  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);
}
)";
  return c;
}

ComputeTaskDescriptor Resize(const OperationDef& definition,
                             const Resize2DAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
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

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.uniform_buffers = {
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
    const uint3 groups_size{8, 8, 1};
    const int dst_layers = DivideRoundUp(dst_shapes[0].c, 4);
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    int groups_z = DivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
