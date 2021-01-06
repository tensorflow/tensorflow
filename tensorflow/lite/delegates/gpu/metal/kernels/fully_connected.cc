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

#include "tensorflow/lite/delegates/gpu/metal/kernels/fully_connected.h"

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetFullyConnectedCode(const GpuInfo& gpu_info, int src_channels,
                                  int dst_channels) {
  bool shared_memory = gpu_info.IsApple() &&
                       gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal();
  const std::string barrier = gpu_info.IsWaveSizeEqualTo32()
                                  ? "SIMDGROUP_BARRIER"
                                  : "threadgroup_barrier";
  const int src_depth = DivideRoundUp(src_channels, 4);
  std::stringstream code;
  code << R"(
    #include <metal_stdlib>
    using namespace metal;

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint3 tid[[thread_position_in_threadgroup]],
                                uint tid_index[[thread_index_in_threadgroup]],
                                uint3 ugid[[thread_position_in_grid]]) {

)";
  if (shared_memory) {
    code << R"(
  float summa = 0.0f;
  threadgroup FLT4 local_vector[32];
  for (int j = 0; j < args.src_depth_sub_groups; ++j) {
    local_vector[tid_index] = j * 32 + tid_index >= args.src_tensor.Slices() ?
      FLT4(0.0f) : args.src_tensor.Read(0, 0, j * 32 + tid_index);
    $0(mem_flags::mem_threadgroup);
    for (uint i = 0, counter = j * 32 + tid.y * 8; i < 8; ++i, ++counter) {
      summa += dot(local_vector[tid.y * 8 + i], args.weights.Read(counter * args.dst_channels_alignedx8 + ugid.x));
    }
    $0(mem_flags::mem_none);
  }
  )";
  } else {
    code << R"(
  float summa = 0.0f;
  int counter = int(ugid.y) * args.src_depth_sub_groups;
  for (int i = 0; i < args.src_depth_sub_groups; ++i, ++counter) {
    )";
    if (src_depth % 4 != 0) {
      code << "    if (counter >= args.src_tensor.Slices()) continue;"
           << std::endl;
    }
    code << "    summa += dot(args.src_tensor.Read(0, 0, counter), "
            "args.weights.Read(counter * "
            "args.dst_channels_alignedx8 + ugid.x));"
         << std::endl;
    code << "  }" << std::endl;
  }
  code << R"(

  threadgroup float temp[8][4];
  temp[tid.x][tid.y] = summa;
  $0(mem_flags::mem_threadgroup);
  if (tid.y == 0) {
    summa += temp[tid.x][1];
    summa += temp[tid.x][2];
    summa += temp[tid.x][3];
    temp[tid.x][0] = summa;
  }
  $0(mem_flags::mem_threadgroup);
  const int linear_index = ugid.x / 4;
  if (tid.y == 0 && tid.x % 4 == 0 && linear_index < args.dst_tensor.Slices()) {
    FLT4 value = FLT4(temp[tid.x][0], temp[tid.x + 1][0], temp[tid.x + 2][0], temp[tid.x + 3][0]) +
      args.bias.Read(linear_index);
    uint3 gid = uint3(0u, 0u, uint(linear_index));
    $$2
    args.dst_tensor.Write(value, 0, 0, linear_index);
  }
}
  )";
  return absl::Substitute(code.str(), barrier);
}
}  // namespace

ComputeTaskDescriptor FullyConnected(const OperationDef& definition,
                                     const FullyConnectedAttributes& attr,
                                     const GpuInfo& gpu_info) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;
  desc.shader_source = GetFullyConnectedCode(gpu_info, attr.weights.shape.i,
                                             attr.weights.shape.o);

  bool shared_memory = gpu_info.IsApple() &&
                       gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal();
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int src_depth_sub_groups = shared_memory ? DivideRoundUp(src_depth, 32)
                                                 : DivideRoundUp(src_depth, 4);
  desc.args.AddInt("dst_channels_alignedx8", AlignByN(attr.weights.shape.o, 8));
  desc.args.AddInt("src_depth_sub_groups", src_depth_sub_groups);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  const int src_depth_aligned = AlignByN(src_depth, shared_memory ? 32 : 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 8);

  int counter = 0;
  std::vector<float> filters_reordered(dst_channels_aligned *
                                       src_depth_aligned * 4);
  for (int j = 0; j < src_depth_aligned; ++j) {
    for (int i = 0; i < dst_channels_aligned; ++i) {
      for (int k = 0; k < 4; ++k) {
        if (j * 4 + k >= attr.weights.shape.i || i >= attr.weights.shape.o) {
          filters_reordered[counter++] = 0.0f;
        } else {
          const int f_index =
              attr.weights.shape.LinearIndex({i, 0, 0, j * 4 + k});
          filters_reordered[counter++] = attr.weights.data[f_index];
        }
      }
    }
  }

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data = GetByteBufferConverted(filters_reordered, data_type);
  weights_desc.size = weights_desc.data.size();

  desc.args.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  BufferDescriptor bias_desc;
  bias_desc.element_type = data_type;
  bias_desc.element_size = 4;
  bias_desc.data = GetByteBufferConvertedResized(attr.bias.data, data_type,
                                                 dst_channels_aligned);
  bias_desc.size = bias_desc.data.size();

  desc.args.AddObject(
      "bias", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  desc.resize_function = [attr](const std::vector<BHWC>& src_shapes,
                                const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
    const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 8);
    int groups_x = DivideRoundUp(dst_channels_aligned, groups_size.x);
    return std::make_pair(groups_size, uint3{groups_x, 1, 1});
  };

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
