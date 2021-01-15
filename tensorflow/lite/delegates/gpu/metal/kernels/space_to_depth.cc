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

ComputeTaskDescriptor SpaceToDepth(const OperationDef& definition,
                                   const SpaceToDepthAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = R"(
kernel void ComputeFunction($0 uint3 gid[[thread_position_in_grid]]) {
  if (gid.x >= args.dst_tensor.Width() || gid.y >= args.dst_tensor.Height() || gid.z >= args.dst_tensor.Slices()) {
    return;
  }
  FLT4 value;
  for (uint i = 0; i < 4; ++i) {
    uint dst_c = 4 * gid.z + i;
    uint block_id = dst_c / args.src_tensor.Channels();
    uint src_x = gid.x * args.block_size + block_id % args.block_size;
    uint src_y = gid.y * args.block_size + block_id / args.block_size;
    uint src_c = dst_c % args.src_tensor.Channels();
    value[i] = args.src_tensor.Read(src_x, src_y, src_c / 4)[src_c % 4];
  }
  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);
})";

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("block_size", attr.block_size);

  desc.resize_function =
      [](const std::vector<BHWC>& src_shapes,
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
