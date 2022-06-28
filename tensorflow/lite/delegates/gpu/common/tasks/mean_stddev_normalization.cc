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

#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"

#include <algorithm>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {

std::string GetReduceCode(const GpuInfo& gpu_info, int reduction_size) {
  // If it is supported, use the built-in work_group_reduce_add function.
  // Otherwise, implement a reduction using __local memory.

  // In the reduction step add upper half of the still-to-be-summed vector to
  // the lower half, while taking care of odd sizes and rounding. E.g.:
  // Number of items still to be summed before: 5
  // Local memory before: [a, b, c, d, e];
  // Local memory after: [a+d, b+e, c, d, e];
  // Threads doing work: id < 2 = floor(5/2)
  // Offset to the added items: 3 = ceil(5/2)
  // Number of items still to be summed after: 3 = ceil(5/2)
  std::string result;
  if (gpu_info.IsApiOpenCl()) {
    result += R"(
#if (__OPENCL_C_VERSION__ >= 200) && (__OPENCL_C_VERSION__ < 300) && \
  !defined(__opencl_c_work_group_collective_functions)
  #define __opencl_c_work_group_collective_functions 1
#endif
)";
  }
  result += R"(
#ifdef __opencl_c_work_group_collective_functions
#define local_reduce(item, tmp, local_id) work_group_reduce_add(item)
#else  // !defined(__opencl_c_work_group_collective_functions)
)";
  if (gpu_info.IsGlsl()) {
    result += "float local_reduce(float item, int local_id) {\n";
  } else {
    result +=
        "float local_reduce(float item, __local float* shared_mem, int "
        "local_id) {\n";
  }
  result += R"(
  shared_mem[local_id] = item;
  LOCAL_MEM_BARRIER;
  // The number of items still need to be summed
)";
  result += "  int reduction_size = " + std::to_string(reduction_size) + ";\n";
  result += R"(  while (reduction_size > 1) {
    int active_thread_limit = reduction_size / 2;
    int offset = (reduction_size + 1) / 2;
    if (local_id < active_thread_limit) {
      item += shared_mem[local_id + offset];
      shared_mem[local_id] = item;
    }
    LOCAL_MEM_BARRIER;
    reduction_size = offset;
  }
  return shared_mem[0];
}
#endif  // defined(__opencl_c_work_group_collective_functions)
)";
  return result;
}

std::string GetFilterCode(const GpuInfo& gpu_info) {
  if (gpu_info.IsGlsl()) {
    return R"(
vec4 filter_outside_tensor(vec4 x, int num_channels, int slice) {
  vec4 result;
  result.x = slice * 4 + 0 < num_channels ? x.x : 0.0f;
  result.y = slice * 4 + 1 < num_channels ? x.y : 0.0f;
  result.z = slice * 4 + 2 < num_channels ? x.z : 0.0f;
  result.w = slice * 4 + 3 < num_channels ? x.w : 0.0f;
  return result;
}
)";
  } else {
    return R"(
float4 filter_outside_tensor(float4 x, int num_channels, int slice) {
  return select(x, INIT_FLOAT4(0.0f), slice * 4 + INIT_INT4v4(0, 1, 2, 3) >= num_channels);
}
)";
  }
}
}  // namespace

MeanStdDevNormalization::MeanStdDevNormalization(const OperationDef& definition,
                                                 const GpuInfo& gpu_info,
                                                 const int tensor_channels)
    : GPUOperation(definition) {
  // The kernel code does not inherently need a fixed size, but in order to not
  // hardcode the __local array's size for the reductions, we would need to pass
  // that size to the kernel at runtime, and that is currently not supported.
  // For now, fix workgroup size to the biggest supported by the device, but not
  // larger than the number of tensor slices.
  const int tensor_slices = DivideRoundUp(tensor_channels, 4);
  int desired_work_group_size =
      std::min(tensor_slices, gpu_info.GetMaxWorkGroupSizeForX());
  if (gpu_info.IsMali()) {
    // Don't use more than 64 work items per work group on ARM Mali. They
    // implement local memory using the global memory, larger workgroups have
    // severe performance penalty.
    desired_work_group_size = 64;
  }
  if (gpu_info.IsAdreno()) {
    AdrenoInfo info = gpu_info.adreno_info;
    if (info.IsAdreno3xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno320 ||
          info.adreno_gpu == AdrenoGpu::kAdreno330) {
        desired_work_group_size = 128;
      } else {
        desired_work_group_size = 64;
      }
    } else if (info.IsAdreno4xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno430) {
        desired_work_group_size = 256;
      } else {
        desired_work_group_size = 128;
      }
    } else if (info.IsAdreno5xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno530 ||
          info.adreno_gpu == AdrenoGpu::kAdreno540) {
        desired_work_group_size = 256;
      } else {
        desired_work_group_size = 128;
      }
    }
  }
  if (gpu_info.IsPowerVR()) {
    desired_work_group_size = 64;
  }
  if (gpu_info.IsApple()) {
    desired_work_group_size = 64;
  }
  while (desired_work_group_size >= tensor_slices * 2) {
    desired_work_group_size /= 2;
  }
  work_group_size_.x = desired_work_group_size;
  work_group_size_.y = 1;  // Required
  work_group_size_.z = 1;  // Required
  args_.AddFloat("variance_bias", 1.0e-8f);
  code_ = GetNormalizationCode(gpu_info, tensor_channels % 4 == 0);
  if (gpu_info.IsCL30OrHigher()) {
    compiler_options_.push_back(CompilerOptions::kCl30);
  } else if (gpu_info.IsCL20OrHigher()) {
    compiler_options_.push_back(CompilerOptions::kCl20);
  }
}

std::string MeanStdDevNormalization::GetNormalizationCode(
    const GpuInfo& gpu_info, bool channels_x4) {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);

  std::string c;
  if (gpu_info.IsGlsl()) {
    c += "shared float shared_mem[" + std::to_string(work_group_size_.x) +
         "];\n";
  }
  c += GetReduceCode(gpu_info, work_group_size_.x);
  if (!channels_x4) {
    c += GetFilterCode(gpu_info);
  }
  if (gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(work_group_size_.x) + ", 1, 1)))\n";
  }
  if (gpu_info.IsGlsl()) {
    c += "#define LOCAL_REDUCE(item, shared_mem, local_id) local_reduce(item, "
         "local_id)\n";
  } else {
    c += "#define LOCAL_REDUCE(item, shared_mem, local_id) local_reduce(item, "
         "shared_mem, local_id)\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  if (!gpu_info.IsGlsl()) {
    c += "#ifndef __opencl_c_work_group_collective_functions\n";
    c += "  __local float tmp[" + std::to_string(work_group_size_.x) + "];\n";
    c += "#endif\n";
  }
  if (definition_.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int B = GLOBAL_ID_1;\n";
    c += "  if (B >= args.src_tensor.Batch()) { return; }\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  c += R"(
  // Calculate the total sum of the input tensor.
  // First, get a local sum of input[local_id_x + N*local_size_x] for all N.
  float4 private_sum4 = INIT_FLOAT4(0.0f);
  int local_id = LOCAL_ID_0;
  for (int S = local_id; S < args.src_tensor.Slices(); S += GROUP_SIZE_0) {
    float4 t = args.src_tensor.Read<float>(0, 0, S);)";
  if (!channels_x4) {
    c += "    t = filter_outside_tensor(t, args.src_tensor.Channels(), S);\n";
  }
  c += R"(
    private_sum4 += t;
  }
  // Reduce the vector to a single float and do a workgroup reduce.
  float private_sum = dot(private_sum4, INIT_FLOAT4(1.0f));
  float sum = LOCAL_REDUCE(private_sum, tmp, local_id);
  // Calculate the mean
  float mean = sum / INIT_FLOAT(args.src_tensor.Channels());
  // Calculate the squared sum of the difference from the mean.
  float4 private_sum_diff_sq4 = INIT_FLOAT4(0.0f);
  for (int S = local_id; S < args.src_tensor.Slices(); S += GROUP_SIZE_0) {
    float4 t = args.src_tensor.Read<float>(0, 0, S);
    float4 diff = t - mean;)";
  if (!channels_x4) {
    c += "    diff = filter_outside_tensor(diff, args.src_tensor.Channels(), "
         "S);\n";
  }
  c += R"(
    private_sum_diff_sq4 += diff * diff;
  }
  // Reduce
  float private_sum_diff_sq = dot(private_sum_diff_sq4, INIT_FLOAT4(1.0f));
  float sum_diff_sq = LOCAL_REDUCE(private_sum_diff_sq, tmp, local_id);
  // Calculate 1/stddev (with the 'regulazing constant' as in tensor_utils.cc)
  float variance = sum_diff_sq / INIT_FLOAT(args.src_tensor.Channels());
  float stddev_inv = rsqrt(variance + args.variance_bias);
  // Calculate (t-mean)/stddev for each element
  for (int S = local_id; S < args.src_tensor.Slices(); S += GROUP_SIZE_0) {
    float4 t = args.src_tensor.Read<float>(0, 0, S);
    FLT4 result = TO_FLT4((t - mean) * stddev_inv);
    args.dst_tensor.Write(result, 0, 0, S);
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
    const OperationDef& definition, const GpuInfo& gpu_info,
    const int tensor_channels) {
  return MeanStdDevNormalization(definition, gpu_info, tensor_channels);
}

}  // namespace gpu
}  // namespace tflite
