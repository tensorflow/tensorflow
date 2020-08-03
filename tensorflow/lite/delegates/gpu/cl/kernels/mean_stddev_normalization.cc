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

#include "tensorflow/lite/delegates/gpu/cl/kernels/mean_stddev_normalization.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetVectorReduceCode() {
  return R"(static inline float reduce_vector(float4 v) {
  return dot(v, (float4)(1.0f));
})";
}

std::string GetReduceCode(size_t work_group_size_x, size_t work_group_size_y) {
  // If it is supported, use the built-in work_group_reduce_add function.
  // Otherwise, implement a reduction using __local memory. Note this only works
  // with power-of-two work group sizes.
  return R"(
static inline float local_reduce(float input) {
#if (__OPENCL_C_VERSION__ >= 300 && __opencl_c_work_group_collective_functions) || \
    (__OPENCL_C_VERSION__ >= 200)
  return work_group_reduce_add(input);
#else
  __local float data[)" +
         std::to_string(work_group_size_y) + "][" +
         std::to_string(work_group_size_x) + R"(];
  const size_t local_id_x = get_local_id(0);
  const size_t local_id_y = get_local_id(1);
  data[local_id_y][local_id_x] = input;
  mem_fence(CLK_LOCAL_MEM_FENCE);
  size_t reduction_size = get_local_size(0) / 2;
  while (reduction_size > 0) {
    if (local_id_x < reduction_size) {
      data[local_id_y][local_id_x] += data[local_id_y][local_id_x + reduction_size];
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    reduction_size /=  2;
  }
  return data[local_id_y][0];
}
#endif
)";
}
}  // namespace

MeanStdDevNormalization::MeanStdDevNormalization(const OperationDef& definition)
    : GPUOperation(definition) {
  // The kernel code does not inherently need a fixed size, but in order to not
  // hardcode the __local array's size for the reductions, we would need to pass
  // that size to the kernel at runtime, and that is currently not supported.
  // For now, fix workgroup size to 128 threads.
  work_group_size_.x = 128;
  work_group_size_.y = 1;
  work_group_size_.z = 1;
  code_ = GetNormalizationCode();
}

std::string MeanStdDevNormalization::GetNormalizationCode() {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);

  std::string c = GetCommonDefines(definition_.precision);
  c += GetVectorReduceCode();
  c += GetReduceCode(work_group_size_.x, work_group_size_.y);
  c += R"(__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void main_function(
$0) {
  size_t B = get_global_id(1);
  if (get_global_id(2) > 0) { return; }
  if (B >= args.src_tensor.Batch()) { return; }
  // Calculate the total sum of the input tensor.
  // First, get a local sum of input[local_id_x + N*local_size_x] for all N.
  float4 private_sum4 = (float4)(0.0f);
  for (int S = get_local_id(0); S < args.src_tensor.Slices(); S += get_local_size(0)) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    // Filter out reads beyond the end of the tensor.
    const int4 is_after_end_of_tensor = (int4)(0, 1, 2, 3) >= (args.src_tensor.Channels() - S * 4);
    const float4 filtered_t = select(t, (float4)(0.0f), is_after_end_of_tensor);
    private_sum4 += filtered_t;
  }
  // Reduce the vector to a single float and do a workgroup reduce.
  const float private_sum = reduce_vector(private_sum4);
  const float sum = local_reduce(private_sum);
  // Calculate the mean
  const float mean = sum / args.src_tensor.Channels();
  // Calculate the squared sum of the difference from the mean.
  float4 private_sum_diff_sq4 = (float4)(0.0f);
  for (int S = get_local_id(0); S < args.src_tensor.Slices(); S += get_local_size(0)) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    const float4 diff = t - mean;
    // Filter out reads beyond the end of the tensor.
    const int4 is_after_end_of_tensor = (int4)(0, 1, 2, 3) >= (args.src_tensor.Channels() - S * 4);
    const float4 filtered_diff = select(diff, (float4)(0.0f), is_after_end_of_tensor);
    // sum_diff_sq += diff²
    private_sum_diff_sq4 = mad(filtered_diff, filtered_diff, private_sum_diff_sq4);
  }
  // Reduce
  const float private_sum_diff_sq = reduce_vector(private_sum_diff_sq4);
  const float sum_diff_sq = local_reduce(private_sum_diff_sq);
  // Calculate 1/stddev (with the 'regulazing constant' as in tensor_utils.cc)
  const float variance = sum_diff_sq / args.src_tensor.Channels();
  const float stddev_inv =  rsqrt(variance + 1.0e-8f);
  // Calculate (t-mean)/stddev for each element
  for (int S = 0; S < args.src_tensor.Slices(); ++S) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    FLT4 result = TO_FLT4((t - mean) * stddev_inv);
    args.dst_tensor.Write(result, 0, 0, S, B);
  }
})";
  return c;
}

int3 MeanStdDevNormalization::GetGridSize() const {
  // To avoid dealing with global reductions, we restrict the grid size to the
  // work group size in the first dimension.
  const int grid_x = work_group_size_.x;
  const int grid_y = src_[0]->Batch();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition) {
  return MeanStdDevNormalization(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
