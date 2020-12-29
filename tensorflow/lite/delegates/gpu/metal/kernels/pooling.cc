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

#include "tensorflow/lite/delegates/gpu/metal/kernels/pooling.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetMaxPoolingCode() {
  std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  if (static_cast<int>(gid.x) >= args.dst_tensor.Width() ||
      static_cast<int>(gid.y) >= args.dst_tensor.Height() ||
      static_cast<int>(gid.z) >= args.dst_tensor.Slices()) {
    return;
  }

  FLT4 maximum = FLT4(-10000.0);
  for (int ky = 0; ky < args.kernel_size_y; ++ky) {
    for (int kx = 0; kx < args.kernel_size_x; ++kx) {
      int c_x = int(gid.x) * args.stride_x - args.offset_x + kx;
      int c_y = int(gid.y) * args.stride_y - args.offset_y + ky;
      bool outside = c_x < 0 || c_y < 0 || c_x >= args.src_tensor.Width() ||
        c_y >= args.src_tensor.Height();
      FLT4 src_color = outside ? FLT4(-10000.0) : args.src_tensor.Read(c_x, c_y, gid.z);
      maximum = max(maximum, src_color);
    }
  }
  args.dst_tensor.GetAddress(linear_index, gid.x, gid.y, gid.z);
  FLT4 value = maximum;
  $2
  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);
}
  )";
  return shader_source;
}

std::string GetMaxPoolingIndicesCode() {
  std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  if (static_cast<int>(gid.x) >= args.dst_tensor.Width() ||
      static_cast<int>(gid.y) >= args.dst_tensor.Height() ||
      static_cast<int>(gid.z) >= args.dst_tensor.Slices()) {
    return;
  }

  FLT4 maximum = FLT4(-10000.0);
  ushort4 indexes = ushort4(0);
  ushort index_counter = 0;
  for (int ky = 0; ky < args.kernel_size_y; ++ky) {
    for (int kx = 0; kx < args.kernel_size_x; ++kx) {
      int c_x = int(gid.x) * args.stride_x - args.offset_x + kx;
      int c_y = int(gid.y) * args.stride_y - args.offset_y + ky;
      bool outside = c_x < 0 || c_y < 0 || c_x >= args.src_tensor.Width() ||
        c_y >= args.src_tensor.Height();
      FLT4 src_color = outside ? FLT4(-10000.0) : args.src_tensor.Read(c_x, c_y, gid.z);
      if (src_color.x > maximum.x) {
        indexes.x = index_counter;
        maximum.x = src_color.x;
      }
      if (src_color.y > maximum.y) {
        indexes.y = index_counter;
        maximum.y = src_color.y;
      }
      if (src_color.z > maximum.z) {
        indexes.z = index_counter;
        maximum.z = src_color.z;
      }
      if (src_color.w > maximum.w) {
        indexes.w = index_counter;
        maximum.w = src_color.w;
      }
      index_counter++;
    }
  }
  args.dst_tensor.GetAddress(linear_index, gid.x, gid.y, gid.z);
  FLT4 value = static_cast<FLT4>(indexes);
  $2
  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);
}
  )";
  return shader_source;
}

std::string GetAveragePoolingCode() {
  std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint tid[[thread_index_in_threadgroup]],
                            uint3 gid[[thread_position_in_grid]]) {
  if (static_cast<int>(gid.x) >= args.dst_tensor.Width() ||
      static_cast<int>(gid.y) >= args.dst_tensor.Height() ||
      static_cast<int>(gid.z) >= args.dst_tensor.Slices()) {
    return;
  }

  float4 sum = float4(0.0f);
  float window_size = 0.0f;
  for (int ky = 0; ky < args.kernel_size_y; ++ky) {
    for (int kx = 0; kx < args.kernel_size_x; ++kx) {
      int c_x = int(gid.x) * args.stride_x - args.offset_x + kx;
      int c_y = int(gid.y) * args.stride_y - args.offset_y + ky;
      bool outside = c_x < 0 || c_y < 0 || c_x >= args.src_tensor.Width() ||
        c_y >= args.src_tensor.Height();
      float4 src_color = outside ? float4(0.0f) : float4(args.src_tensor.Read(c_x, c_y, gid.z));
      window_size += outside ? 0.0f : 1.0f;
      sum += src_color;
    }
  }
  args.dst_tensor.GetAddress(linear_index, gid.x, gid.y, gid.z);
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  FLT4 value = FLT4(sum / window_size);
  $2
  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);
}
)";
  return shader_source;
}

}  // namespace

ComputeTaskDescriptor Pooling(const OperationDef& definition,
                              const Pooling2DAttributes& attr,
                              bool generate_indices) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  if (attr.type == PoolingType::MAX) {
    desc.shader_source =
        generate_indices ? GetMaxPoolingIndicesCode() : GetMaxPoolingCode();
  } else if (attr.type == PoolingType::AVERAGE) {
    desc.shader_source = GetAveragePoolingCode();
  }

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("kernel_size_x", attr.kernel.w);
  desc.args.AddInt("kernel_size_y", attr.kernel.h);
  desc.args.AddInt("stride_x", attr.strides.w);
  desc.args.AddInt("stride_y", attr.strides.h);
  desc.args.AddInt("offset_x", attr.padding.prepended.w);
  desc.args.AddInt("offset_y", attr.padding.prepended.h);

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
