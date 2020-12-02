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

#include "tensorflow/lite/delegates/gpu/metal/kernels/space_to_depth.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {

ComputeTaskDescriptor SpaceToDepth(ValueId input_id, ValueId output_id,
                                   const SpaceToDepthAttributes& attr) {
  ComputeTaskDescriptor desc;
  desc.shader_source = R"(
#include <metal_stdlib>
using namespace metal;
struct uniforms {
  uint4 src_size;
  uint4 dst_size;
  uint4 block_size;
};
$0
kernel void ComputeFunction($1 uint3 gid[[thread_position_in_grid]]) {
  uint3 src_size = (uint3)(params.src_size.xyz);
  uint3 dst_size = (uint3)(params.dst_size.xyz);
  uint block_size = (uint)(params.block_size.x);
  if (gid.x >= dst_size.x || gid.y >= dst_size.y || gid.z * 4 >= dst_size.z) {
    return;
  }
  FLT4 value;
  for (uint i = 0; i < 4; ++i) {
    uint dst_c = 4 * gid.z + i;
    uint block_id = dst_c / src_size.z;
    uint src_x = gid.x * block_size + block_id % block_size;
    uint src_y = gid.y * block_size + block_id / block_size;
    uint src_c = dst_c % src_size.z;
    value[i] =
        src_buffer[src_x + src_size.x * (src_y + src_size.y * (src_c / 4))]
                  [src_c % 4];
  }
  $2
  dst_buffer[gid.x + dst_size.x * (gid.y + dst_size.y * gid.z)] = value;
})";

  desc.AddSrcTensor("src_buffer");
  desc.AddDstTensor("dst_buffer");

  desc.uniform_buffers = {
      {"constant uniforms& params",
       [attr](const std::vector<BHWC>& src_shapes,
              const std::vector<BHWC>& dst_shapes) {
         const std::vector<int> uniform_params = {
             // src_size
             src_shapes[0].w,
             src_shapes[0].h,
             src_shapes[0].c,
             0,
             // dst_size
             dst_shapes[0].w,
             dst_shapes[0].h,
             dst_shapes[0].c,
             0,
             // block_size
             attr.block_size,
             0,
             0,
             0,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc.resize_function =
      [attr](const std::vector<BHWC>& src_shapes,
             const std::vector<BHWC>& dst_shapes) -> std::pair<uint3, uint3> {
    const uint3 grid = uint3(dst_shapes[0].w, dst_shapes[0].h,
                             DivideRoundUp(dst_shapes[0].c, 4));
    const uint3 groups_size = GetWorkGroupSizeForGrid(grid);
    const int groups_x = DivideRoundUp(grid.x, groups_size.x);
    const int groups_y = DivideRoundUp(grid.y, groups_size.y);
    const int groups_z = DivideRoundUp(grid.z, groups_size.z);
    return std::make_pair(groups_size, uint3(groups_x, groups_y, groups_z));
  };
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
