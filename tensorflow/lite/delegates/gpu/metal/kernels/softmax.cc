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
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
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
$0
kernel void ComputeFunction($1
                            uint tid[[thread_index_in_threadgroup]],
                            uint3 ugid[[thread_position_in_grid]])
{

  float4 maxx4 = float4(args.src_tensor.Read(0, 0, 0).x);
  for (int s = int(tid); s < args.src_tensor.Slices(); s += 32) {
    float4 mask_a = s == args.src_tensor.Slices() - 1 ? float4(args.mask_x, args.mask_y, args.mask_z, args.mask_w) : float4(1.0f);
    float4 mask_b = float4(1.0f) - mask_a;
    float4 src = float4(args.src_tensor.Read(0, 0, s));
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
  for (int s = int(tid); s < args.src_tensor.Slices(); s += 32) {
    float4 mask_temp = s == args.src_tensor.Slices() - 1 ? float4(args.mask_x, args.mask_y, args.mask_z, args.mask_w) : float4(1.0f);
    float4 src = float4(args.src_tensor.Read(0, 0, s)) - float4(maximum);
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
  if (dst_s < args.src_tensor.Slices()) {
    float4 src = float4(args.src_tensor.Read(0, 0, dst_s)) - float4(maximum);
    FLT4 value = FLT4(exp(src) * sum);
    uint3 gid = uint3(0, 0, dst_s);
    $2
    args.dst_tensor.Write(value, 0, 0, dst_s);
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
$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  if (int(gid.x) >= args.dst_tensor.Width() || int(gid.y) >= args.dst_tensor.Height()) {
    return;
  }

  float maximum = args.src_tensor.Read(gid.x, gid.y, 0).x;
  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {
    float4 mask_a = d == args.dst_tensor.Slices() - 1 ? float4(args.mask_x, args.mask_y, args.mask_z, args.mask_w) : float4(1.0f);
    float4 mask_b = float4(1.0f) - mask_a;
    float4 src = float4(args.src_tensor.Read(gid.x, gid.y, d));
    src = src * mask_a + mask_b * src.x;
    maximum = max(maximum, src.x);
    maximum = max(maximum, src.y);
    maximum = max(maximum, src.z);
    maximum = max(maximum, src.w);
  }

  float sum = 0.0f;
  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {
    float4 mask_temp = d == args.dst_tensor.Slices() - 1 ? float4(args.mask_x, args.mask_y, args.mask_z, args.mask_w) : float4(1.0f);
    float4 src = float4(args.src_tensor.Read(gid.x, gid.y, d)) - float4(maximum);
    sum += dot(mask_temp, exp(src));
  }

  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {
    float4 src = float4(args.src_tensor.Read(gid.x, gid.y, d)) - float4(maximum);
    FLT4 value = FLT4(exp(src) / sum);
    $2
    args.dst_tensor.Write(value, gid.x, gid.y, d);
  }
}
  )";

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddFloat("mask_x");
  desc.args.AddFloat("mask_y");
  desc.args.AddFloat("mask_z");
  desc.args.AddFloat("mask_w");

  desc.update_function = {[](const std::vector<BHWC>& src_shapes,
                             const std::vector<BHWC>& dst_shapes,
                             ArgumentsBinder* args) -> absl::Status {
    float4 mask = GetMaskForLastPlane(dst_shapes[0].c);
    RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
    RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
    RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
    RETURN_IF_ERROR(args->SetFloat("mask_w", mask.w));
    return absl::OkStatus();
  }};

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

  desc.args.AddFloat("mask_x");
  desc.args.AddFloat("mask_y");
  desc.args.AddFloat("mask_z");
  desc.args.AddFloat("mask_w");

  desc.update_function = {[](const std::vector<BHWC>& src_shapes,
                             const std::vector<BHWC>& dst_shapes,
                             ArgumentsBinder* args) -> absl::Status {
    float4 mask = GetMaskForLastPlane(dst_shapes[0].c);
    RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
    RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
    RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
    RETURN_IF_ERROR(args->SetFloat("mask_w", mask.w));
    return absl::OkStatus();
  }};

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
