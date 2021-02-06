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
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
std::string GetKernelDepthWiseConv3x3Stride2() {
  std::string code = R"(
kernel void ComputeFunction($0
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

  r0 += TO_ACCUM_TYPE(s0 * filters_loc[0]);
  r0 += TO_ACCUM_TYPE(s1 * filters_loc[1]);
  r0 += TO_ACCUM_TYPE(s2 * filters_loc[2]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y1)] * FLT(!(x0_out || y1_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y1)] * FLT(!(x1_out || y1_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y1)] * FLT(!(x2_out || y1_out));

  r0 += TO_ACCUM_TYPE(s0 * filters_loc[3]);
  r0 += TO_ACCUM_TYPE(s1 * filters_loc[4]);
  r0 += TO_ACCUM_TYPE(s2 * filters_loc[5]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y2)] * FLT(!(x0_out || y2_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y2)] * FLT(!(x1_out || y2_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y2)] * FLT(!(x2_out || y2_out));

  r0 += TO_ACCUM_TYPE(s0 * filters_loc[6]);
  r0 += TO_ACCUM_TYPE(s1 * filters_loc[7]);
  r0 += TO_ACCUM_TYPE(s2 * filters_loc[8]);
  l0 += TO_ACCUM_TYPE(s0 * filters_loc[0]);
  l0 += TO_ACCUM_TYPE(s1 * filters_loc[1]);
  l0 += TO_ACCUM_TYPE(s2 * filters_loc[2]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y3)] * FLT(!(x0_out || y3_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y3)] * FLT(!(x1_out || y3_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y3)] * FLT(!(x2_out || y3_out));

  l0 += TO_ACCUM_TYPE(s0 * filters_loc[3]);
  l0 += TO_ACCUM_TYPE(s1 * filters_loc[4]);
  l0 += TO_ACCUM_TYPE(s2 * filters_loc[5]);

  s0 = src_loc[args.src_tensor.GetWHOffset(x0, y4)] * FLT(!(x0_out || y4_out));
  s1 = src_loc[args.src_tensor.GetWHOffset(x1, y4)] * FLT(!(x1_out || y4_out));
  s2 = src_loc[args.src_tensor.GetWHOffset(x2, y4)] * FLT(!(x2_out || y4_out));

  l0 += TO_ACCUM_TYPE(s0 * filters_loc[6]);
  l0 += TO_ACCUM_TYPE(s1 * filters_loc[7]);
  l0 += TO_ACCUM_TYPE(s2 * filters_loc[8]);

  r0 += TO_ACCUM_TYPE(filters_loc[9]);
  l0 += TO_ACCUM_TYPE(filters_loc[9]);

  bool y0_in = gid_y < args.dst_tensor.Height();
  bool y1_in = gid_y + 1 < args.dst_tensor.Height();

  if (y0_in) {
    FLT4 value = FLT4(r0);
    args.dst_tensor.Write(value, gid_x, gid_y, gid_z);
  }
  if (y1_in) {
    FLT4 value = FLT4(l0);
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

int3 DepthWiseConv3x3Stride2::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

DepthWiseConv3x3Stride2 CreateDepthWiseConv3x3Stride2(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  DepthWiseConv3x3Stride2 desc(definition);
  desc.code_ = GetKernelDepthWiseConv3x3Stride2();
  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args_.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args_.AddInt("padding_y", -attr.padding.prepended.h);
  desc.args_.AddInt("stride_x", attr.strides.w);
  desc.args_.AddInt("dilation_x", attr.dilations.w);

  // For this operation we keep weights and biases in one buffer
  auto weights_reordered = ReorderWeightsDepthWiseConv3x3Stride2(attr);
  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data = GetByteBufferConverted(weights_reordered, data_type);
  weights_desc.size = weights_desc.data.size();
  desc.args_.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  desc.work_group_size_ = int3(8, 4, 1);
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
