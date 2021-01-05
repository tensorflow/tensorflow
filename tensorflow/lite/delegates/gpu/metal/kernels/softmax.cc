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

#include "tensorflow/lite/delegates/gpu/metal/kernels/softmax.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
std::string GetSoftmax1x1Code(const GpuInfo& gpu_info) {
  const std::string barrier = gpu_info.IsWaveSizeEqualTo32()
                                  ? "SIMDGROUP_BARRIER"
                                  : "threadgroup_barrier";
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 size;
  float4 mask;
};

$0

kernel void ComputeFunction($1
                            uint tid[[thread_index_in_threadgroup]],
                            uint3 ugid[[thread_position_in_grid]])
{

  float4 maxx4 = float4(src_tensor[0].x);
  for (int s = int(tid); s < params.size.x; s += 32) {
    float4 mask_a = s == params.size.x - 1 ? params.mask : float4(1.0f);
    float4 mask_b = float4(1.0f) - mask_a;
    float4 src = float4(src_tensor[s]);
    src = src * mask_a + mask_b * src.x;
    maxx4 = max(maxx4, src);
  }
  float maximum = max(maxx4.x, maxx4.y);
  maximum = max(maximum, maxx4.z);
  maximum = max(maximum, maxx4.w);

  threadgroup float4 tmp[8];
  threadgroup float* tmpx1 = (threadgroup float*)tmp;

  tmpx1[tid] = maximum;
)";
  code += "  " + barrier + "(mem_flags::mem_threadgroup);\n";
  code += R"(
  if (tid == 0) {
    maxx4 = max(tmp[0], tmp[1]);
    maxx4 = max(maxx4, tmp[2]);
    maxx4 = max(maxx4, tmp[3]);
    maxx4 = max(maxx4, tmp[4]);
    maxx4 = max(maxx4, tmp[5]);
    maxx4 = max(maxx4, tmp[6]);
    maxx4 = max(maxx4, tmp[7]);
    maximum = max(maxx4.x, maxx4.y);
    maximum = max(maximum, maxx4.z);
    maximum = max(maximum, maxx4.w);
    tmpx1[0] = maximum;
  }
)";
  code += "  " + barrier + "(mem_flags::mem_threadgroup);\n";
  code += R"(
  maximum = tmpx1[0];

  float sum = 0.0f;
  for (int s = int(tid); s < params.size.x; s += 32) {
    float4 mask_temp = s == params.size.x - 1 ? params.mask : float4(1.0f);
    float4 src = float4(src_tensor[s]) - float4(maximum);
    sum += dot(mask_temp, exp(src));
  }

)";
  code += "  " + barrier + "(mem_flags::mem_threadgroup);\n";
  code += R"(

  tmpx1[tid] = sum;
)";
  code += "  " + barrier + "(mem_flags::mem_threadgroup);\n";
  code += R"(
  if (tid == 0) {
    sum = dot(float4(1.0f), tmp[0]);
    sum += dot(float4(1.0f), tmp[1]);
    sum += dot(float4(1.0f), tmp[2]);
    sum += dot(float4(1.0f), tmp[3]);
    sum += dot(float4(1.0f), tmp[4]);
    sum += dot(float4(1.0f), tmp[5]);
    sum += dot(float4(1.0f), tmp[6]);
    sum += dot(float4(1.0f), tmp[7]);
    tmpx1[0] = 1.0 / sum;
  }
)";
  code += "  " + barrier + "(mem_flags::mem_threadgroup);\n";
  code += R"(
  sum = tmpx1[0];

  int dst_s = int(ugid.x);
  if (dst_s < params.size.x) {
    int linear_index = dst_s;
    float4 src = float4(src_tensor[linear_index]) - float4(maximum);
    FLT4 value = FLT4(exp(src) * sum);
    uint3 gid = uint3(0, 0, linear_index);
    $2
    dst_tensor[linear_index] = value;
  }
})";
  return code;
}
}  // namespace

ComputeTaskDescriptor Softmax(const OperationDef& definition) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 size;
  float4 mask;
};
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  if (int(gid.x) >= params.size.x || int(gid.y) >= params.size.y) {
    return;
  }

  float maximum = src_tensor[gid.y * params.size.x + gid.x].x;
  for (int d = 0; d < params.size.z; ++d) {
    int buffer_index = (d * params.size.y + gid.y) * params.size.x + gid.x;
    float4 mask_a = d == params.size.z - 1 ? params.mask : float4(1.0f);
    float4 mask_b = float4(1.0f) - mask_a;
    float4 src = float4(src_tensor[buffer_index]);
    src = src * mask_a + mask_b * src.x;
    maximum = max(maximum, src.x);
    maximum = max(maximum, src.y);
    maximum = max(maximum, src.z);
    maximum = max(maximum, src.w);
  }

  float sum = 0.0f;
  for (int d = 0; d < params.size.z; ++d) {
    int buffer_index = (d * params.size.y + gid.y) * params.size.x + gid.x;
    float4 mask_temp = d == params.size.z - 1 ? params.mask : float4(1.0f);
    float4 src = float4(src_tensor[buffer_index]) - float4(maximum);
    sum += dot(mask_temp, exp(src));
  }

  for (int d = 0; d < params.size.z; ++d) {
    const int linear_index = (d * params.size.y + gid.y) * params.size.x + gid.x;
    float4 src = float4(src_tensor[linear_index]) - float4(maximum);
    FLT4 value = FLT4(exp(src) / sum);
    $2
    dst_tensor[linear_index] = value;
  }
}
  )";

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.uniform_buffers = {
      {"constant uniforms& params",
       [](const std::vector<BHWC>& src_shapes,
          const std::vector<BHWC>& dst_shapes) {
         const int dst_depth = DivideRoundUp(dst_shapes[0].c, 4);
         struct uniforms {
           int4 size;
           float4 mask;
         };
         uniforms params;
         params.size = {dst_shapes[0].w, dst_shapes[0].h, dst_depth, 1};
         params.mask = {0.0f, 0.0f, 0.0f, 0.0f};
         int reminder = dst_shapes[0].c % 4 == 0 ? 4 : dst_shapes[0].c % 4;
         for (int i = 0; i < reminder; ++i) {
           params.mask[i] = 1.0f;
         }
         const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&params);
         return std::vector<uint8_t>(ptr, ptr + sizeof(uniforms));
       }},
  };

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    uint3 groups_size{8, 4, 1};
    uint3 groups_count{DivideRoundUp(dst_shapes[0].w, groups_size.x),
                       DivideRoundUp(dst_shapes[0].h, groups_size.y), 1};
    return std::make_pair(groups_size, groups_count);
  };

  return desc;
}

ComputeTaskDescriptor Softmax1x1(const OperationDef& definition,
                                 const GpuInfo& gpu_info) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetSoftmax1x1Code(gpu_info);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.uniform_buffers = {
      {"constant uniforms& params",
       [](const std::vector<BHWC>& src_shapes,
          const std::vector<BHWC>& dst_shapes) {
         const int src_depth = DivideRoundUp(dst_shapes[0].c, 4);
         struct uniforms {
           int4 size;
           float4 mask;
         };
         uniforms params;
         params.size = {src_depth, DivideRoundUp(src_depth, 32), 1, 1};
         params.mask = {0.0f, 0.0f, 0.0f, 0.0f};
         int reminder = dst_shapes[0].c % 4 == 0 ? 4 : dst_shapes[0].c % 4;
         for (int i = 0; i < reminder; ++i) {
           params.mask[i] = 1.0f;
         }
         const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&params);
         return std::vector<uint8_t>(ptr, ptr + sizeof(uniforms));
       }},
  };

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    uint3 groups_size{32, 1, 1};
    uint3 groups_count{
        DivideRoundUp(DivideRoundUp(dst_shapes[0].c, 4), groups_size.x), 1, 1};
    return std::make_pair(groups_size, groups_count);
  };

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
