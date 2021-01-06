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

#include "tensorflow/lite/delegates/gpu/metal/kernels/reshape.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
std::string GetReshapeCode() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  const int3 igid = int3(gid);

  if (igid.x >= args.dst_tensor.Width() || igid.y >= args.dst_tensor.Height() ||
      igid.z >= args.dst_tensor.Slices()) return;

  FLT4 value;

  for (int i = 0; i < 4; ++i) {
    const int dst_channel = igid.z * 4 + i;
    if (dst_channel < args.dst_tensor.Channels()) {
      int p = (igid.y * args.dst_tensor.Width() + igid.x) * args.dst_tensor.Channels() + dst_channel;
      int src_wc = args.src_tensor.Width() * args.src_tensor.Channels();
      int src_y = p / src_wc;
      int t0 = p - src_y * src_wc;  // p % src_wc;
      int src_x = t0 / args.src_tensor.Channels();
      int src_z = t0 - src_x * args.src_tensor.Channels();  // t0 % args.src_tensor.Channels();
      int src_layer = src_z >> 2;
      int src_channel = src_z & 3;
      value[i] = args.src_tensor.Read(src_x, src_y, src_layer)[src_channel];
    }
  }

  $2
  args.dst_tensor.Write(value, igid.x, igid.y, igid.z);
})";
  return code;
}

std::string GetReshapex4Code() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  int X = gid.x;
  int Y = gid.y;
  int Z = gid.z;

  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;

  int p = (Y * args.dst_tensor.Width() + X) * args.dst_tensor.Slices() + Z;
  int src_ws = args.src_tensor.Width() * args.src_tensor.Slices();
  int src_y = p / src_ws;
  int t0 = p - src_y * src_ws;  // p % src_ws;
  int src_x = t0 / args.src_tensor.Slices();
  int src_z = t0 - src_x * args.src_tensor.Slices();  // t0 % args.src_tensor.Slices();

  FLT4 value = args.src_tensor.Read(src_x, src_y, src_z);
  $2
  args.dst_tensor.Write(value, X, Y, Z);
})";
  return code;
}

}  // namespace

ComputeTaskDescriptor Reshape(const OperationDef& definition) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetReshapeCode();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 grid = uint3(dst_shapes[0].w, dst_shapes[0].h,
                             DivideRoundUp(dst_shapes[0].c, 4));
    const uint3 groups_size = GetWorkGroupSizeForGrid(grid);
    int groups_x = DivideRoundUp(grid.x, groups_size.x);
    int groups_y = DivideRoundUp(grid.y, groups_size.y);
    int groups_z = DivideRoundUp(grid.z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

ComputeTaskDescriptor Reshapex4(const OperationDef& definition) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetReshapex4Code();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 grid = uint3(dst_shapes[0].w, dst_shapes[0].h,
                             DivideRoundUp(dst_shapes[0].c, 4));
    const uint3 groups_size = GetWorkGroupSizeForGrid(grid);
    int groups_x = DivideRoundUp(grid.x, groups_size.x);
    int groups_y = DivideRoundUp(grid.y, groups_size.y);
    int groups_z = DivideRoundUp(grid.z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
