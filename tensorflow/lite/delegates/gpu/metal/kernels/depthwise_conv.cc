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

#include "tensorflow/lite/delegates/gpu/metal/kernels/depthwise_conv.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetKernelDepthWiseConv3x3Stride1x1() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

$0

kernel void ComputeFunction(
                            $1
                            uint3 ugid[[thread_position_in_grid]])
{
  int gid_x = ugid.x * 2;
  int gid_y = ugid.y * 2;
  int gid_z = ugid.z;

  if (gid_x >= args.dst_tensor.Width() || gid_y >= args.dst_tensor.Height()) {
    return;
  }

  ACCUM_FLT4 r0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 l0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 t0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 b0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);

  int x0 = gid_x + args.padding_x;
  int x1 = gid_x + args.padding_x + 1;
  int x2 = gid_x + args.padding_x + 2;
  int x3 = gid_x + args.padding_x + 3;
  int y0 = gid_y + args.padding_y;
  int y1 = gid_y + args.padding_y + 1;
  int y2 = gid_y + args.padding_y + 2;
  int y3 = gid_y + args.padding_y + 3;

  bool x0_out = x0 < 0 || x0 >= args.src_tensor.Width();
  bool x1_out = x1 < 0 || x1 >= args.src_tensor.Width();
  bool x2_out = x2 < 0 || x2 >= args.src_tensor.Width();
  bool x3_out = x3 < 0 || x3 >= args.src_tensor.Width();
  bool y0_out = y0 < 0 || y0 >= args.src_tensor.Height();
  bool y1_out = y1 < 0 || y1 >= args.src_tensor.Height();
  bool y2_out = y2 < 0 || y2 >= args.src_tensor.Height();
  bool y3_out = y3 < 0 || y3 >= args.src_tensor.Height();

  x0 = clamp(x0, 0, args.src_tensor.Width() - 1);
  x1 = clamp(x1, 0, args.src_tensor.Width() - 1);
  x2 = clamp(x2, 0, args.src_tensor.Width() - 1);
  x3 = clamp(x3, 0, args.src_tensor.Width() - 1);
  y0 = clamp(y0, 0, args.src_tensor.Height() - 1);
  y1 = clamp(y1, 0, args.src_tensor.Height() - 1);
  y2 = clamp(y2, 0, args.src_tensor.Height() - 1);
  y3 = clamp(y3, 0, args.src_tensor.Height() - 1);

  device FLT4* src_loc = args.src_tensor.GetPtrWithSliceOffset(gid_z);
  device FLT4* filters_loc = args.weights.GetPtr() + gid_z * 10;

  FLT4 s0 = src_loc[args.src_tensor.GetWHOffset(x0, y0)] * FLT(!(x0_out || y0_out));
  FLT4 s1 = src_loc[args.src_tensor.GetWHOffset(x0, y1)] * FLT(!(x0_out || y1_out));
  FLT4 s2 = src_loc[args.src_tensor.GetWHOffset(x0, y2)] * FLT(!(x0_out || y2_out));
  FLT4 s3 = src_loc[args.src_tensor.GetWHOffset(x0, y3)] * FLT(!(x0_out || y3_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[0]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[1]);
  l0 += TO_ACCUM4_TYPE(s3 * filters_loc[2]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x1, y0)] * FLT(!(x1_out || y0_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y1)] * FLT(!(x1_out || y1_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x1, y2)] * FLT(!(x1_out || y2_out));
  s3 = src_loc[args.src_tensor.GetWHOffset(x1, y3)] * FLT(!(x1_out || y3_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[3]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[4]);
  l0 += TO_ACCUM4_TYPE(s3 * filters_loc[5]);
  t0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
  t0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
  t0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);
  b0 += TO_ACCUM4_TYPE(s1 * filters_loc[0]);
  b0 += TO_ACCUM4_TYPE(s2 * filters_loc[1]);
  b0 += TO_ACCUM4_TYPE(s3 * filters_loc[2]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x2, y0)] * FLT(!(x2_out || y0_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x2, y1)] * FLT(!(x2_out || y1_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y2)] * FLT(!(x2_out || y2_out));
  s3 = src_loc[args.src_tensor.GetWHOffset(x2, y3)] * FLT(!(x2_out || y3_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[6]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[7]);
  l0 += TO_ACCUM4_TYPE(s3 * filters_loc[8]);
  t0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
  t0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
  t0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);
  b0 += TO_ACCUM4_TYPE(s1 * filters_loc[3]);
  b0 += TO_ACCUM4_TYPE(s2 * filters_loc[4]);
  b0 += TO_ACCUM4_TYPE(s3 * filters_loc[5]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x3, y0)] * FLT(!(x3_out || y0_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x3, y1)] * FLT(!(x3_out || y1_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x3, y2)] * FLT(!(x3_out || y2_out));
  s3 = src_loc[args.src_tensor.GetWHOffset(x3, y3)] * FLT(!(x3_out || y3_out));

  t0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
  t0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
  t0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);
  b0 += TO_ACCUM4_TYPE(s1 * filters_loc[6]);
  b0 += TO_ACCUM4_TYPE(s2 * filters_loc[7]);
  b0 += TO_ACCUM4_TYPE(s3 * filters_loc[8]);

  r0 += TO_ACCUM4_TYPE(filters_loc[9]);
  l0 += TO_ACCUM4_TYPE(filters_loc[9]);
  t0 += TO_ACCUM4_TYPE(filters_loc[9]);
  b0 += TO_ACCUM4_TYPE(filters_loc[9]);

  bool x0_in = gid_x < args.dst_tensor.Width();
  bool x1_in = gid_x + 1 < args.dst_tensor.Width();
  bool y0_in = gid_y < args.dst_tensor.Height();
  bool y1_in = gid_y + 1 < args.dst_tensor.Height();

  if (y0_in && x0_in) {
    args.dst_tensor.GetAddress(linear_index, gid_x, gid_y, gid_z);
    FLT4 value = FLT4(r0);
    uint3 gid = uint3(gid_x, gid_y, gid_z);
    $2
    args.dst_tensor.Write(value, gid_x, gid_y, gid_z);
  }
  if (y1_in && x0_in) {
    args.dst_tensor.GetAddress(linear_index, gid_x, gid_y + 1, gid_z);
    FLT4 value = FLT4(l0);
    uint3 gid = uint3(gid_x, gid_y + 1, gid_z);
    $2
    args.dst_tensor.Write(value, gid_x, gid_y + 1, gid_z);
  }
  if (y0_in && x1_in) {
    args.dst_tensor.GetAddress(linear_index, gid_x + 1, gid_y, gid_z);
    FLT4 value = FLT4(t0);
    uint3 gid = uint3(gid_x + 1, gid_y, gid_z);
    $2
    args.dst_tensor.Write(value, gid_x + 1, gid_y, gid_z);
  }
  if (y1_in && x1_in) {
    args.dst_tensor.GetAddress(linear_index, gid_x + 1, gid_y + 1, gid_z);
    FLT4 value = FLT4(b0);
    uint3 gid = uint3(gid_x + 1, gid_y + 1, gid_z);
    $2
    args.dst_tensor.Write(value, gid_x + 1, gid_y + 1, gid_z);
  }
}
  )";

  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// DepthWiseConv3x3Stride1x1
std::vector<float> ReorderWeightsDepthWiseConv3x3Stride1x1(
    const DepthwiseConvolution2DAttributes& attr) {
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int kernel_x = 3;
  const int kernel_y = 3;
  std::vector<float> weights_reordered((kernel_x * kernel_y + 1) * src_depth *
                                       4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int x = 0; x < kernel_x; ++x) {
      for (int y = 0; y < kernel_y; ++y) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < attr.weights.shape.i) {
            const int f_index = attr.weights.shape.LinearIndex({0, y, x, s_ch});
            weights_reordered[counter++] = attr.weights.data[f_index];
          } else {
            weights_reordered[counter++] = 0.0f;
          }
        }
      }
    }

    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      if (dst_ch < attr.bias.shape.v) {
        weights_reordered[counter++] = attr.bias.data[dst_ch];
      } else {
        weights_reordered[counter++] = 0.0f;
      }
    }
  }

  return weights_reordered;
}

std::string GetKernelDepthWiseConv3x3Stride2() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

$0

kernel void ComputeFunction(
                            $1
                            uint3 ugid[[thread_position_in_grid]])
{
  int gid_x = ugid.x;
  int gid_y = ugid.y * 2;
  int gid_z = ugid.z;

  if (gid_x >= args.dst_tensor.Width() || gid_y >= args.dst_tensor.Height()) {
    return;
  }

  device FLT4* src_loc = args.src_tensor.GetPtrWithSliceOffset(gid_z);
  device FLT4* filters_loc = args.weights.GetPtr() + gid_z * 10;

  ACCUM_FLT4 r0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 l0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);

  int x0 = gid_x * args.stride_x + args.padding_x;
  int x1 = gid_x * args.stride_x + args.padding_x + args.dilation_x;
  int x2 = gid_x * args.stride_x + args.padding_x + 2 * args.dilation_x;
  int y0 = gid_y * 2 + args.padding_y;
  int y1 = gid_y * 2 + args.padding_y + 1;
  int y2 = gid_y * 2 + args.padding_y + 2;
  int y3 = gid_y * 2 + args.padding_y + 3;
  int y4 = gid_y * 2 + args.padding_y + 4;

  bool x0_out = x0 < 0 || x0 >= args.src_tensor.Width();
  bool x1_out = x1 < 0 || x1 >= args.src_tensor.Width();
  bool x2_out = x2 < 0 || x2 >= args.src_tensor.Width();
  bool y0_out = y0 < 0 || y0 >= args.src_tensor.Height();
  bool y1_out = y1 < 0 || y1 >= args.src_tensor.Height();
  bool y2_out = y2 < 0 || y2 >= args.src_tensor.Height();
  bool y3_out = y3 < 0 || y3 >= args.src_tensor.Height();
  bool y4_out = y4 < 0 || y4 >= args.src_tensor.Height();

  x0 = clamp(x0, 0, args.src_tensor.Width() - 1);
  x1 = clamp(x1, 0, args.src_tensor.Width() - 1);
  x2 = clamp(x2, 0, args.src_tensor.Width() - 1);
  y0 = clamp(y0, 0, args.src_tensor.Height() - 1);
  y1 = clamp(y1, 0, args.src_tensor.Height() - 1);
  y2 = clamp(y2, 0, args.src_tensor.Height() - 1);
  y3 = clamp(y3, 0, args.src_tensor.Height() - 1);
  y4 = clamp(y4, 0, args.src_tensor.Height() - 1);

  FLT4 s0 = src_loc[args.src_tensor.GetWHOffset(x0, y0)] * FLT(!(x0_out || y0_out));
  FLT4 s1 = src_loc[args.src_tensor.GetWHOffset(x1, y0)] * FLT(!(x1_out || y0_out));
  FLT4 s2 = src_loc[args.src_tensor.GetWHOffset(x2, y0)] * FLT(!(x2_out || y0_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y1)] * FLT(!(x0_out || y1_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y1)] * FLT(!(x1_out || y1_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y1)] * FLT(!(x2_out || y1_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y2)] * FLT(!(x0_out || y2_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y2)] * FLT(!(x1_out || y2_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y2)] * FLT(!(x2_out || y2_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);
  l0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y3)] * FLT(!(x0_out || y3_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y3)] * FLT(!(x1_out || y3_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y3)] * FLT(!(x2_out || y3_out));

  l0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y4)] * FLT(!(x0_out || y4_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y4)] * FLT(!(x1_out || y4_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y4)] * FLT(!(x2_out || y4_out));

  l0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);

  r0 += TO_ACCUM4_TYPE(filters_loc[9]);
  l0 += TO_ACCUM4_TYPE(filters_loc[9]);

  bool y0_in = gid_y < args.dst_tensor.Height();
  bool y1_in = gid_y + 1 < args.dst_tensor.Height();

  if (y0_in) {
    args.dst_tensor.GetAddress(linear_index, gid_x, gid_y, gid_z);
    FLT4 value = FLT4(r0);
    uint3 gid = uint3(gid_x, gid_y, gid_z);
    $2
    args.dst_tensor.Write(value, gid_x, gid_y, gid_z);
  }
  if (y1_in) {
    args.dst_tensor.GetAddress(linear_index, gid_x, gid_y + 1, gid_z);
    FLT4 value = FLT4(l0);
    uint3 gid = uint3(gid_x, gid_y + 1, gid_z);
    $2
    args.dst_tensor.Write(value, gid_x, gid_y + 1, gid_z);
  }
}
  )";

  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// DepthWiseConv3x3Stride2
std::vector<float> ReorderWeightsDepthWiseConv3x3Stride2(
    const DepthwiseConvolution2DAttributes& attr) {
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int kernel_x = 3;
  const int kernel_y = 3;
  std::vector<float> weights_reordered((kernel_x * kernel_y + 1) * src_depth *
                                       4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < attr.weights.shape.i) {
            const int f_index = attr.weights.shape.LinearIndex({0, y, x, s_ch});
            weights_reordered[counter++] = attr.weights.data[f_index];
          } else {
            weights_reordered[counter++] = 0.0f;
          }
        }
      }
    }

    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      if (dst_ch < attr.bias.shape.v) {
        weights_reordered[counter++] = attr.bias.data[dst_ch];
      } else {
        weights_reordered[counter++] = 0.0f;
      }
    }
  }

  return weights_reordered;
}

}  // namespace

ComputeTaskDescriptor DepthWiseConvolution(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  int channels_multiplier = attr.weights.shape.o;
  std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;
$0
kernel void ComputeFunction(
                            $1
                            uint tid[[thread_index_in_threadgroup]],
                            uint3 gid[[thread_position_in_grid]]) {
  int dst_x = static_cast<int>(gid.x);
  int dst_y = static_cast<int>(gid.y);
  int dst_z = static_cast<int>(gid.z);

  if (dst_x >= args.dst_tensor.Width() || dst_y >= args.dst_tensor.Height()) return;

  device FLT4* temp = args.weights.GetPtr() + dst_z * args.kernel_size_x * args.kernel_size_y;
  ACCUM_FLT4 sum0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);

  int src_x = dst_x * args.stride_x + args.padding_x;
  int src_y = dst_y * args.stride_y + args.padding_y;

  for(int ky = 0; ky < args.kernel_size_y; ++ky) {
    int yc = ky * args.dilation_y + src_y;
    if (yc < 0 || yc >= args.src_tensor.Height()) continue;
    for(int kx = 0; kx < args.kernel_size_x; ++kx) {
      int xc = kx * args.dilation_x + src_x;
      if (xc < 0 || xc >= args.src_tensor.Width()) continue;
)";
  if (channels_multiplier == 1) {
    shader_source += R"(
      int src_layer = dst_z;
      FLT4 src_modified = args.src_tensor.Read(xc, yc, src_layer);
)";
  } else if (channels_multiplier == 2) {
    shader_source += R"(
      int src_layer = dst_z / 2;
      FLT4 src = args.src_tensor.Read(xc, yc, src_layer);
      FLT2 t0 = dst_z % 2 == 0 ? src.xy : src.zw;
      FLT4 src_modified = FLT4(t0.x, t0.x, t0.y, t0.y);
)";
  } else if (channels_multiplier == 4) {
    shader_source += R"(
      int src_layer = dst_z / 4;
      FLT4 src = args.src_tensor.Read(xc, yc, src_layer);
      FLT t0 = src[dst_z % 4];
      FLT4 src_modified = FLT4(t0, t0, t0, t0);
)";
  } else {
    shader_source += R"(
      int src_layer = dst_z / args.channel_multiplier;
      FLT4 src = args.src_tensor.Read(xc, yc, src_layer);
      FLT4 src_modified;
      const int src_layer_offset = (dst_z % args.channel_multiplier) * 4;
      src_modified.x = src[(src_layer_offset + 0) / args.channel_multiplier];
      src_modified.y = src[(src_layer_offset + 1) / args.channel_multiplier];
      src_modified.z = src[(src_layer_offset + 2) / args.channel_multiplier];
      src_modified.w = src[(src_layer_offset + 3) / args.channel_multiplier];
)";
  }
  shader_source += R"(
      sum0 += TO_ACCUM4_TYPE(src_modified * temp[ky * args.kernel_size_x + kx]);
    }
  }
  FLT4 res = FLT4(sum0) + args.biases.Read(dst_z);
  args.dst_tensor.GetAddress(linear_index, dst_x, dst_y, dst_z);
  FLT4 value = res;
  $2
  args.dst_tensor.Write(value, dst_x, dst_y, dst_z);
}
)";
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = shader_source;
  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args.AddInt("padding_y", -attr.padding.prepended.h);
  desc.args.AddInt("dilation_x", attr.dilations.w);
  desc.args.AddInt("dilation_y", attr.dilations.h);
  desc.args.AddInt("stride_x", attr.strides.w);
  desc.args.AddInt("stride_y", attr.strides.h);
  desc.args.AddInt("kernel_size_x", attr.weights.shape.w);
  desc.args.AddInt("kernel_size_y", attr.weights.shape.h);
  desc.args.AddInt("channel_multiplier", attr.weights.shape.o);

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  const int output_channels_count = attr.weights.shape.i * attr.weights.shape.o;
  const int dst_ch_aligned = AlignByN(output_channels_count, 4);
  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data =
      GetByteBufferConverted(ConvertToPIOHW4(attr.weights), data_type);
  weights_desc.size = weights_desc.data.size();
  desc.args.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  BufferDescriptor bias_desc;
  bias_desc.element_type = data_type;
  bias_desc.element_size = 4;
  bias_desc.data =
      GetByteBufferConvertedResized(attr.bias.data, data_type, dst_ch_aligned);
  bias_desc.size = bias_desc.data.size();
  desc.args.AddObject(
      "biases", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    uint3 groups_size{8, 4, 1};
    uint3 groups_count{DivideRoundUp(dst_shapes[0].w, groups_size.x),
                       DivideRoundUp(dst_shapes[0].h, groups_size.y),
                       DivideRoundUp(dst_shapes[0].c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return desc;
}

ComputeTaskDescriptor DepthWiseConv3x3Stride1x1(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetKernelDepthWiseConv3x3Stride1x1();
  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args.AddInt("padding_y", -attr.padding.prepended.h);

  // For this operation we keep weights and biases in one buffer
  auto weights_reordered = ReorderWeightsDepthWiseConv3x3Stride1x1(attr);
  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data = GetByteBufferConverted(weights_reordered, data_type);
  weights_desc.size = weights_desc.data.size();
  desc.args.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const int grid_x = DivideRoundUp(dst_shapes[0].w, 2);
    const int grid_y = DivideRoundUp(dst_shapes[0].h, 2);
    const int grid_z = DivideRoundUp(dst_shapes[0].c, 4);
    uint3 group_size{8, 4, 1};
    if (grid_x <= 4) {
      group_size.x = 4;
      group_size.z = grid_z % 2 == 0 ? 2 : 1;
    }
    const int groups_x = DivideRoundUp(grid_x, group_size.x);
    const int groups_y = DivideRoundUp(grid_y, group_size.y);
    const int groups_z = DivideRoundUp(grid_z, group_size.z);
    return std::make_pair(group_size, uint3(groups_x, groups_y, groups_z));
  };

  return desc;
}

bool CheckDepthWiseConv3x3Stride1x1Support(
    const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.weights.shape.h == 3 &&
         attr.weights.shape.w == 3 && attr.strides.h == 1 &&
         attr.strides.w == 1 && attr.dilations.h == 1 && attr.dilations.w == 1;
}

ComputeTaskDescriptor DepthWiseConv3x3Stride2(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetKernelDepthWiseConv3x3Stride2();
  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args.AddInt("padding_y", -attr.padding.prepended.h);
  desc.args.AddInt("stride_x", attr.strides.w);
  desc.args.AddInt("dilation_x", attr.dilations.w);

  // For this operation we keep weights and biases in one buffer
  auto weights_reordered = ReorderWeightsDepthWiseConv3x3Stride2(attr);
  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data = GetByteBufferConverted(weights_reordered, data_type);
  weights_desc.size = weights_desc.data.size();
  desc.args.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const int grid_x = dst_shapes[0].w;
    const int grid_y = DivideRoundUp(dst_shapes[0].h, 2);
    const int grid_z = DivideRoundUp(dst_shapes[0].c, 4);
    const uint3 group_size{8, 4, 1};
    const int groups_x = DivideRoundUp(grid_x, group_size.x);
    const int groups_y = DivideRoundUp(grid_y, group_size.y);
    const int groups_z = DivideRoundUp(grid_z, group_size.z);
    return std::make_pair(group_size, uint3(groups_x, groups_y, groups_z));
  };

  return desc;
}

bool CheckDepthWiseConv3x3Stride2Support(
    const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.weights.shape.h == 3 &&
         attr.weights.shape.w == 3 && attr.strides.h == 2 &&
         attr.dilations.h == 1;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
