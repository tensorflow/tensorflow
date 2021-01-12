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

#include "tensorflow/lite/delegates/gpu/metal/kernels/mean.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {

std::string GetMeanCode(const int3& work_group_size) {
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  std::string c = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint tid[[thread_index_in_threadgroup]],
                                uint3 tid3d[[thread_position_in_threadgroup]],
                                uint3 gid[[thread_position_in_grid]]) {
  int local_x = static_cast<int>(tid3d.x);
  int local_y = static_cast<int>(tid3d.y);
  int local_id = static_cast<int>(tid);
  int S = static_cast<int>(gid.z);
  if (S >= args.dst_tensor.Slices()) return;
)";
  c += "  threadgroup float4 accum[" +
       std::to_string(work_group_size.x * work_group_size.y) + "];\n";
  c += "  accum[local_id] = float4(0.0f);\n";
  c += "  for (int s_y = local_y; s_y < args.src_tensor.Height(); s_y += " +
       wg_y + ") {\n";
  c += "    for (int s_x = local_x; s_x < args.src_tensor.Width(); s_x += " +
       wg_x + ") {\n";
  c += "      accum[local_id] += float4(args.src_tensor.Read(s_x, s_y, S));\n";
  c += "    }\n";
  c += "  }\n";
  c += "  accum[local_id] *= args.inv_multiplier_x;\n";
  c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  const int total_size = work_group_size.x * work_group_size.y;
  int offset = 1;
  int reminder = total_size / 4;
  for (; reminder >= 8; reminder /= 4, offset *= 4) {
    c += "  if (local_id < " + std::to_string(reminder) + ") {\n";
    c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
    c += "    float4 sum = accum[t + " + std::to_string(offset) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 2) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 3) + "];\n";
    c += "    accum[t] += sum;\n";
    c += "  }\n";
    c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  c += "  float4 sum = accum[0];\n";
  reminder *= 4;
  for (int i = 1; i < reminder; ++i) {
    c += "  sum += accum[" + std::to_string(offset * i) + "];\n";
  }
  c += "  FLT4 value = FLT4(sum * args.inv_multiplier_y);\n";
  c += R"(
  args.dst_tensor.Write(value, 0, 0, gid.z);
}
)";
  return c;
}

ComputeTaskDescriptor Mean(const OperationDef& definition,
                           const MeanAttributes& attr) {
  if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
    // Mean calculation is supported only for height and width
    return {};
  }

  const int3 work_group_size = int3(16, 16, 1);

  ComputeTaskDescriptor desc(definition);
  std::string code = GetMeanCode(work_group_size);
  desc.shader_source = code;

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddFloat("inv_multiplier_x");
  desc.args.AddFloat("inv_multiplier_y");

  desc.update_function = {
      [work_group_size](const std::vector<BHWC>& src_shapes,
                        const std::vector<BHWC>& dst_shapes,
                        ArgumentsBinder* args) -> absl::Status {
        const double total_size = src_shapes[0].w * src_shapes[0].h;
        const double size_0 = work_group_size.x * work_group_size.y;
        const double size_1 = total_size / size_0;
        RETURN_IF_ERROR(args->SetFloat("inv_multiplier_x", 1.0 / size_1));
        RETURN_IF_ERROR(args->SetFloat("inv_multiplier_y", 1.0 / size_0));
        return absl::OkStatus();
      }};

  desc.resize_function = [work_group_size](
                             const std::vector<BHWC>& src_shapes,
                             const std::vector<BHWC>& dst_shapes) {
    const int dst_slices = DivideRoundUp(dst_shapes[0].c, 4);
    const int groups_z = DivideRoundUp(dst_slices, work_group_size.z);
    return std::make_pair(work_group_size, uint3{1, 1, groups_z});
  };
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
