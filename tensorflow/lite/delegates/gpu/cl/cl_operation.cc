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

#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
int3 GetWorkGroupsCount(int grid_dimension, const int3& grid_size,
                        const int3& work_group_size,
                        const int3& work_group_launch_order) {
  int3 work_groups_count;
  if (grid_dimension == 1) {
    work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
    work_groups_count.y = 1;
    work_groups_count.z = 1;
  } else if (grid_dimension == 2) {
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = 1;
  } else {  // grid_dimension == 3
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    wgs.z = DivideRoundUp(grid_size.z, work_group_size.z);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = wgs[work_group_launch_order[2]];
  }
  return work_groups_count;
}

std::string GetCommonOpenCLDefines(CalculationsPrecision precision) {
  std::string result;

  result += "#define FLT16_0123(V) V.s0123\n";
  result += "#define FLT16_4567(V) V.s4567\n";
  result += "#define FLT16_89ab(V) V.s89ab\n";
  result += "#define FLT16_cdef(V) V.scdef\n";
  result += "#define GLOBAL_ID_0 get_global_id(0)\n";
  result += "#define GLOBAL_ID_1 get_global_id(1)\n";
  result += "#define GLOBAL_ID_2 get_global_id(2)\n";
  result += "#define LOCAL_ID_0 get_local_id(0)\n";
  result += "#define LOCAL_ID_1 get_local_id(1)\n";
  result += "#define LOCAL_ID_2 get_local_id(2)\n";
  result += "#define GROUP_ID_0 get_group_id(0)\n";
  result += "#define GROUP_ID_1 get_group_id(1)\n";
  result += "#define GROUP_ID_2 get_group_id(2)\n";
  result += "#define GROUP_SIZE_0 get_local_size(0)\n";
  result += "#define GROUP_SIZE_1 get_local_size(1)\n";
  result += "#define GROUP_SIZE_2 get_local_size(2)\n";
  result += "#define SUB_GROUP_LOCAL_ID get_sub_group_local_id()\n";
  result += "#define SUB_GROUP_BROADCAST(V, ID) sub_group_broadcast(V, ID)\n";
  result += "#define SIMD_LOCAL_MEM_BARRIER barrier(CLK_LOCAL_MEM_FENCE)\n";
  result += "#define LOCAL_MEM_BARRIER barrier(CLK_LOCAL_MEM_FENCE)\n";
  result += "#define MAIN_FUNCTION __kernel void main_function\n";
  result += "#define INIT_FLOAT(value) (float)(value)\n";
  result += "#define INIT_FLOAT2(value) (float2)(value)\n";
  result += "#define INIT_FLOAT2v2(v0, v1) (float2)(v0, v1)\n";
  result += "#define INIT_FLOAT3(value) (float3)(value)\n";
  result += "#define INIT_FLOAT3v3(v0, v1, v2) (float3)(v0, v1, v2)\n";
  result += "#define INIT_FLOAT4(value) (float4)(value)\n";
  result += "#define INIT_FLOAT4v4(v0, v1, v2, v3) (float4)(v0, v1, v2, v3)\n";
  result += "#define INIT_INT(value) (int)(value)\n";
  result += "#define INIT_INT2v2(v0, v1) (int2)(v0, v1)\n";
  result += "#define INIT_INT4v4(v0, v1, v2, v3) (int4)(v0, v1, v2, v3)\n";
  result += "#define CONVERT_TO_INT4(value) convert_int4(value)\n";
  switch (precision) {
    case CalculationsPrecision::F32:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define INIT_ACCUM_FLT4(value) (float4)(value)\n";
      result += "#define FLT float\n";
      result += "#define FLT2 float2\n";
      result += "#define FLT3 float3\n";
      result += "#define FLT4 float4\n";
      result += "#define TO_FLT4 convert_float4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      result += "#define TO_ACCUM_FLT2 convert_float2\n";
      result += "#define TO_ACCUM_FLT3 convert_float3\n";
      result += "#define TO_ACCUM_FLT4 convert_float4\n";
      result += "#define INIT_FLT(value) (float)(value)\n";
      result += "#define INIT_FLT4(value) (float4)(value)\n";
      result +=
          "#define INIT_FLT4v4(v0, v1, v2, v3) (float4)(v0, v1, v2, v3)\n";
      break;
    case CalculationsPrecision::F16:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 half4\n";
      result += "#define INIT_ACCUM_FLT4(value) (half4)(value)\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_half4\n";
      result += "#define TO_ACCUM_FLT convert_half\n";
      result += "#define TO_ACCUM_FLT2 convert_half2\n";
      result += "#define TO_ACCUM_FLT3 convert_half3\n";
      result += "#define TO_ACCUM_FLT4 convert_half4\n";
      result += "#define INIT_FLT(value) (half)(value)\n";
      result += "#define INIT_FLT4(value) (half4)(value)\n";
      result += "#define INIT_FLT4v4(v0, v1, v2, v3) (half4)(v0, v1, v2, v3)\n";
      break;
    case CalculationsPrecision::F32_F16:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define INIT_ACCUM_FLT4(value) (float4)(value)\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      result += "#define TO_ACCUM_FLT2 convert_float2\n";
      result += "#define TO_ACCUM_FLT3 convert_float3\n";
      result += "#define TO_ACCUM_FLT4 convert_float4\n";
      result += "#define INIT_FLT(value) (half)(value)\n";
      result += "#define INIT_FLT4(value) (half4)(value)\n";
      result += "#define INIT_FLT4v4(v0, v1, v2, v3) (half4)(v0, v1, v2, v3)\n";
      break;
  }
  return result;
}
}  // namespace

ClOperation::ClOperation(ClOperation&& operation)
    : operation_(std::move(operation.operation_)),
      kernel_(std::move(operation.kernel_)),
      cl_args_(std::move(operation.cl_args_)) {}

ClOperation& ClOperation::operator=(ClOperation&& operation) {
  if (this != &operation) {
    operation_ = std::move(operation.operation_);
    kernel_ = std::move(operation.kernel_);
    cl_args_ = std::move(operation.cl_args_);
  }
  return *this;
}

absl::Status ClOperation::AddOperation(ClOperation* operation) {
  return operation_->AddOperation(operation->operation_.get());
}

absl::Status ClOperation::UpdateParams() {
  for (int i = 0; i < operation_->src_tensors_names_.size(); ++i) {
    const auto* cl_spatial_tensor =
        dynamic_cast<const Tensor*>(operation_->src_[i]);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    RETURN_IF_ERROR(cl_args_.SetObjectRef(operation_->src_tensors_names_[i],
                                          cl_spatial_tensor));
  }
  for (int i = 0; i < operation_->dst_tensors_names_.size(); ++i) {
    const auto* cl_spatial_tensor =
        dynamic_cast<const Tensor*>(operation_->dst_[i]);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    RETURN_IF_ERROR(cl_args_.SetObjectRef(operation_->dst_tensors_names_[i],
                                          cl_spatial_tensor));
  }
  RETURN_IF_ERROR(operation_->BindArguments(&cl_args_));
  operation_->grid_size_ = operation_->GetGridSize();
  operation_->work_groups_count_ = GetWorkGroupsCount(
      operation_->grid_dimension_, operation_->grid_size_,
      operation_->work_group_size_, operation_->work_group_launch_order_);
  return absl::OkStatus();
}

absl::Status ClOperation::Compile(const CreationContext& creation_context) {
  operation_->AssembleCode(creation_context.GetGpuInfo());
  operation_->code_ =
      GetCommonOpenCLDefines(operation_->definition_.precision) +
      operation_->code_;
  RETURN_IF_ERROR(cl_args_.Init(
      creation_context.GetGpuInfo(),
      {{operation_->dst_tensors_names_[0], operation_->elementwise_code_}},
      creation_context.context, &operation_->args_, &operation_->code_));
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      operation_->code_, "main_function", operation_->compiler_options_,
      *creation_context.context, *creation_context.device, &kernel_));
  return operation_->PostCompileCheck(creation_context.GetGpuInfo(),
                                      kernel_.info_);
}

absl::Status ClOperation::CompileDeserialized(
    const CreationContext& creation_context) {
  RETURN_IF_ERROR(cl_args_.Init(creation_context.GetGpuInfo(),
                                &operation_->args_, creation_context.context));
  return creation_context.cache->GetOrCreateCLKernel(
      operation_->code_, "main_function", operation_->compiler_options_,
      *creation_context.context, *creation_context.device, &kernel_);
}

absl::Status ClOperation::Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                               ProfilingCommandQueue* profiling_queue) {
  std::vector<int3> possible_work_groups;
  operation_->GetPossibleKernelWorkGroups(tuning_type, gpu_info, kernel_.info_,
                                          &possible_work_groups);
  if (possible_work_groups.empty()) {
    return absl::NotFoundError(
        "Can not found work_group size to launch kernel");
  }
  if (possible_work_groups.size() == 1) {
    operation_->work_group_size_ = possible_work_groups[0];
    operation_->work_groups_count_ = GetWorkGroupsCount(
        operation_->grid_dimension_, operation_->grid_size_,
        operation_->work_group_size_, operation_->work_group_launch_order_);
    return absl::OkStatus();
  } else {
    std::vector<int3> work_groups_count(possible_work_groups.size());
    for (int i = 0; i < work_groups_count.size(); ++i) {
      work_groups_count[i] = GetWorkGroupsCount(
          operation_->grid_dimension_, operation_->grid_size_,
          possible_work_groups[i], operation_->work_group_launch_order_);
    }
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    int best_work_group_index;
    RETURN_IF_ERROR(profiling_queue->GetBestWorkGroupIndex(
        kernel_, gpu_info, work_groups_count, possible_work_groups,
        &best_work_group_index));
    operation_->work_group_size_ = possible_work_groups[best_work_group_index];
    operation_->work_groups_count_ = GetWorkGroupsCount(
        operation_->grid_dimension_, operation_->grid_size_,
        operation_->work_group_size_, operation_->work_group_launch_order_);
    return absl::OkStatus();
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
