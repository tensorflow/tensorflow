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

namespace {
bool Is4Aligned(const SliceAttributes& attr) {
  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int4 GetOffset(const SliceAttributes& attr, int src_width, int src_height,
               int src_channels, int src_batch) {
  int4 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Aligned(attr)) {
    offset.z /= 4;
  }
  if (attr.strides.b > 0) {
    offset.w = attr.starts.b;
  } else {
    if (attr.ends.b > 0) {
      offset.w = attr.ends.b;
    } else {
      offset.w = src_batch + attr.ends.b;
    }
  }
  return offset;
}

}  // namespace

std::string GetSliceCode(const OperationDef& op_def, bool alignedx4) {
  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  std::string c = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 offset;
  int4 stride;
};

$0
kernel void ComputeFunction($1
                            uint3 gid[[thread_position_in_grid]]) {
)";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = static_cast<int>(gid.x);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = static_cast<int>(gid.x);\n";
  }
  c += "  int Y = static_cast<int>(gid.y);\n";
  c += "  int Z = static_cast<int>(gid.z);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int s_x = X * params.stride.x + params.offset.x;\n";
  c += "  int s_y = Y * params.stride.y + params.offset.y;\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int s_b = " + batch_id + " * params.stride.w + params.offset.w;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }
  if (alignedx4) {
    c += "  int s_z = Z + params.offset.z;\n";
    c += "  FLT4 result = args.src_tensor.Read(s_x, s_y, s_z);\n";
  } else {
    c += "  FLT4 result;\n";
    const std::string postfixes[] = {"x", "y", "z", "w"};
    for (int i = 0; i < 4; ++i) {
      c += "  {\n";
      const std::string ch = "(Z * 4 + " + std::to_string(i) + ")";
      c += "    int s_ch = " + ch + " * params.stride.z + params.offset.z;\n";
      c += "    int s_z = min(s_ch >> 2, args.src_tensor.Slices() - 1);\n";
      c += "    int s_z_rem = s_ch & 3;\n";
      c += "    FLT4 t = args.src_tensor.Read(s_x, s_y, s_z);\n";
      c += "    result." + postfixes[i] + " = t[s_ch & 3];\n";
      c += "  }\n";
    }
  }
  c += "  FLT4 value = result;\n";
  c += "  args.dst_tensor.GetAddress(linear_index, X, Y, Z);\n";
  c += "  $2\n";
  c += "  args.dst_tensor.Write(value, X, Y, Z);\n";
  c += "}\n";
  return c;
}
}  // namespace

ComputeTaskDescriptor Slice(const OperationDef& definition,
                            const SliceAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetSliceCode(definition, Is4Aligned(attr));

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.uniform_buffers = {
      {"constant uniforms& params",
       [attr](const std::vector<BHWC>& src_shapes,
              const std::vector<BHWC>& dst_shapes) {
         int4 offset = GetOffset(attr, src_shapes[0].w, src_shapes[0].h,
                                 src_shapes[0].c, src_shapes[0].b);
         std::vector<int> uniform_params{
             // int4 offset
             offset.x,
             offset.y,
             offset.z,
             offset.w,
             // int4 stride
             attr.strides.w,
             attr.strides.h,
             attr.strides.c,
             attr.strides.b,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc.resize_function = [attr](const std::vector<BHWC>& src_shapes,
                                const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
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
