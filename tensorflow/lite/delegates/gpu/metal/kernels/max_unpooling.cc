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

#include "tensorflow/lite/delegates/gpu/metal/kernels/max_unpooling.h"

#include <map>
#include <memory>
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

std::string GetMaxUnpoolingCode() {
  std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  int X = static_cast<int>(gid.x);
  int Y = static_cast<int>(gid.y);
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) {
    return;
  }

  int src_x = (X + args.offset_x) / args.stride_x;
  int src_y = (Y + args.offset_y) / args.stride_y;

  bool outside = src_x < 0 || src_y < 0 ||
    src_x >= args.src_tensor.Width() || src_y >= args.src_tensor.Height();

  int4 indexes = outside ? int4(0) : int4(args.src_indices.Read(src_x, src_y, gid.z));
  FLT4 src_color = outside ? FLT4(0.0f) : args.src_tensor.Read(src_x, src_y, gid.z);

  int t_x = X - (src_x * args.stride_x - args.offset_x);
  int t_y = Y - (src_y * args.stride_y - args.offset_y);
  int t_index = t_y * args.kernel_size_x + t_x;

  FLT4 value;
  value.x = t_index == indexes.x ? src_color.x : 0.0;
  value.y = t_index == indexes.y ? src_color.y : 0.0;
  value.z = t_index == indexes.z ? src_color.z : 0.0;
  value.w = t_index == indexes.w ? src_color.w : 0.0;

  args.dst_tensor.GetAddress(linear_index, X, Y, gid.z);
  $2
  args.dst_tensor.Write(value, X, Y, gid.z);
}
  )";
  return shader_source;
}
}  // namespace

ComputeTaskDescriptor MaxUnpooling(const OperationDef& definition,
                                   const MaxUnpooling2DAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetMaxUnpoolingCode();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddSrcTensor("src_indices", definition.src_tensors[1]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("kernel_size_x", attr.kernel.w);
  desc.args.AddInt("stride_x", attr.strides.w);
  desc.args.AddInt("stride_y", attr.strides.h);
  desc.args.AddInt("offset_x", attr.padding.prepended.w);
  desc.args.AddInt("offset_y", attr.padding.prepended.h);

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    int groups_z = DivideRoundUp(dst_shapes[0].c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
