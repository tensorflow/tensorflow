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

#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"

#include <Availability.h>

#include <map>
#include <string>
#include <tuple>

#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/kernel_info.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {
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
}  // namespace

void ComputeTask::Init(std::unique_ptr<GPUOperation>&& operation) {
  operation_ = std::move(operation);
}

const OperationDef& ComputeTask::GetDefinition() const {
  return operation_->definition_;
}

bool ComputeTask::IsLinkable() const { return operation_->IsLinkable(); }

absl::Status ComputeTask::AddTask(ComputeTask* task) {
  return operation_->AddOperation(task->operation_.get());
}

absl::Status ComputeTask::Compile(MetalDevice* device) {
  operation_->AssembleCode(device->GetInfo());
  const std::map<std::string, std::string> linkables = {
      {operation_->dst_tensors_names_[0], operation_->elementwise_code_}};
  RETURN_IF_ERROR(metal_args_.Init(linkables, device, &operation_->args_,
                                   &operation_->code_));

  operation_->args_.ReleaseCPURepresentation();

  return CompileProgram(device, operation_->definition_.precision,
                        operation_->code_);
}

absl::Status ComputeTask::CompileProgram(MetalDevice* device,
                                         CalculationsPrecision precision,
                                         const std::string& kernel_code) {
  NSString* barrier;
  // simdgroup_barrier is supported since Metal shading language version 2.0
  if (device->IsLanguageVersion2orHigher()) {
    barrier = @"simdgroup_barrier";
  } else {
    barrier = @"threadgroup_barrier";
  }
  NSString* storageType;
  NSString* accumulatorType;
  NSString* toAccumulatorType4 = @"";
  if (precision == CalculationsPrecision::F32) {
    storageType = @"float";
    accumulatorType = @"float";
  } else {
    // FP16
    storageType = @"half";
    if (precision == CalculationsPrecision::F32_F16) {
      accumulatorType = @"float";
      toAccumulatorType4 = @"float4";
    } else {
      accumulatorType = @"half";
    }
  }
  NSDictionary<NSString*, NSString*>* macros = @{
    @"float16" : @"float4x4",
    @"half16" : @"half4x4",
    @"float8" : @"float2x4",
    @"half8" : @"half2x4",
    @"FLT16_0123(V)" : @"V[0]",
    @"FLT16_4567(V)" : @"V[1]",
    @"FLT16_89ab(V)" : @"V[2]",
    @"FLT16_cdef(V)" : @"V[3]",
    @"FLT" : storageType,
    @"FLT2" : [NSString stringWithFormat:@"%@2", storageType],
    @"FLT3" : [NSString stringWithFormat:@"%@3", storageType],
    @"FLT4" : [NSString stringWithFormat:@"%@4", storageType],
    @"ACCUM_FLT" : accumulatorType,
    @"ACCUM_FLT2" : [NSString stringWithFormat:@"%@2", accumulatorType],
    @"ACCUM_FLT3" : [NSString stringWithFormat:@"%@3", accumulatorType],
    @"ACCUM_FLT4" : [NSString stringWithFormat:@"%@4", accumulatorType],
    @"INIT_ACCUM_FLT4(value)" :
        [NSString stringWithFormat:@"%@4(value)", accumulatorType],
    @"TO_ACCUM_TYPE" : toAccumulatorType4,
    @"TO_ACCUM_FLT" : accumulatorType,
    @"TO_ACCUM_FLT2" : [NSString stringWithFormat:@"%@2", accumulatorType],
    @"TO_ACCUM_FLT3" : [NSString stringWithFormat:@"%@3", accumulatorType],
    @"TO_ACCUM_FLT4" : [NSString stringWithFormat:@"%@4", accumulatorType],
    @"TO_FLT4" : [NSString stringWithFormat:@"%@4", storageType],
    @"SIMDGROUP_BARRIER" : barrier,
    @"SIMD_LOCAL_MEM_BARRIER" : barrier,
    @"MAIN_FUNCTION" : @"\"kernel void ComputeFunction\"",
    @"GLOBAL_ID_0" : @"static_cast<int>(reserved_gid.x)",
    @"GLOBAL_ID_1" : @"static_cast<int>(reserved_gid.y)",
    @"GLOBAL_ID_2" : @"static_cast<int>(reserved_gid.z)",
    @"LOCAL_ID_0" : @"static_cast<int>(reserved_lid.x)",
    @"LOCAL_ID_1" : @"static_cast<int>(reserved_lid.y)",
    @"LOCAL_ID_2" : @"static_cast<int>(reserved_lid.z)",
    @"GROUP_ID_0" : @"static_cast<int>(reserved_group_id.x)",
    @"GROUP_ID_1" : @"static_cast<int>(reserved_group_id.y)",
    @"GROUP_ID_2" : @"static_cast<int>(reserved_group_id.z)",
    @"GROUP_SIZE_0" : @"static_cast<int>(reserved_group_size.x)",
    @"GROUP_SIZE_1" : @"static_cast<int>(reserved_group_size.y)",
    @"GROUP_SIZE_2" : @"static_cast<int>(reserved_group_size.z)",
    @"SUB_GROUP_LOCAL_ID" : @"static_cast<int>(reserved_simd_id)",
    @"\"SUB_GROUP_BROADCAST(V, ID)\"" : @"\"simd_broadcast(V, ID)\"",
    @"__local" : @"threadgroup",
    @"__global" : @"device",
    @"__constant" : @"constant",
    @"LOCAL_MEM_BARRIER" : @"threadgroup_barrier(mem_flags::mem_threadgroup)",
    @"INIT_FLT(value)" : [NSString stringWithFormat:@"%@(value)", storageType],
    @"INIT_FLT4(value)" :
        [NSString stringWithFormat:@"%@4(value)", storageType],
    @"\"INIT_FLT4v4(v0, v1, v2, v3)\"" :
        [NSString stringWithFormat:@"\"%@4(v0, v1, v2, v3)\"", storageType],
    @"INIT_FLOAT(value)" : @"float(value)",
    @"INIT_FLOAT2(value)" : @"float2(value)",
    @"\"INIT_FLOAT2v2(v0, v1)\"" : @"\"float2(v0, v1)\"",
    @"INIT_FLOAT3(value)" : @"float3(value)",
    @"\"INIT_FLOAT3v3(v0, v1, v2)\"" : @"\"float3(v0, v1, v2)\"",
    @"INIT_FLOAT4(value)" : @"float4(value)",
    @"\"INIT_FLOAT4v4(v0, v1, v2, v3)\"" : @"\"float4(v0, v1, v2, v3)\"",
    @"INIT_INT(value)" : @"int(value)",
    @"\"INIT_INT2v2(v0, v1)\"" : @"\"int2(v0, v1)\"",
    @"\"INIT_INT4v4(v0, v1, v2, v3)\"" : @"\"int4(v0, v1, v2, v3)\"",
    @"CONVERT_TO_INT4(value)" : @"int4(value)",
  };

  NSString* code =
      [NSString stringWithCString:kernel_code.c_str()
                         encoding:[NSString defaultCStringEncoding]];
  id<MTLComputePipelineState> program;
  RETURN_IF_ERROR(CreateComputeProgram(device->device(), code,
                                       @"ComputeFunction", macros, &program));
  if (!program) {
    return absl::InternalError("Unknown shader compilation error");
  }
  program_ = program;
  return absl::OkStatus();
}

absl::Status ComputeTask::UpdateParams() {
  for (int i = 0; i < operation_->src_tensors_names_.size(); ++i) {
    const auto* metal_spatial_tensor =
        dynamic_cast<const MetalSpatialTensor*>(operation_->src_[i]);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    RETURN_IF_ERROR(metal_args_.SetObjectRef(operation_->src_tensors_names_[i],
                                             *metal_spatial_tensor));
  }
  for (int i = 0; i < operation_->dst_tensors_names_.size(); ++i) {
    const auto* metal_spatial_tensor =
        dynamic_cast<const MetalSpatialTensor*>(operation_->dst_[i]);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    RETURN_IF_ERROR(metal_args_.SetObjectRef(operation_->dst_tensors_names_[i],
                                             *metal_spatial_tensor));
  }
  RETURN_IF_ERROR(operation_->BindArguments(&metal_args_));
  operation_->grid_size_ = operation_->GetGridSize();
  operation_->work_groups_count_ = GetWorkGroupsCount(
      operation_->grid_dimension_, operation_->grid_size_,
      operation_->work_group_size_, operation_->work_group_launch_order_);
  return absl::OkStatus();
}

void ComputeTask::Encode(id<MTLComputeCommandEncoder> encoder) {
  [encoder setComputePipelineState:program_];
  metal_args_.Encode(encoder, 0);
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->work_groups_count_.x;
  groupsCount.height = operation_->work_groups_count_.y;
  groupsCount.depth = operation_->work_groups_count_.z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

void ComputeTask::SetSrcTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetSrc(tensor, index);
  auto status =
      metal_args_.SetObjectRef(operation_->src_tensors_names_[index], *tensor);
}

void ComputeTask::SetDstTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetDst(tensor, index);
  auto status =
      metal_args_.SetObjectRef(operation_->dst_tensors_names_[index], *tensor);
}

absl::Status ComputeTask::Tune(TuningType tuning_type, MetalDevice* device) {
  std::vector<int3> possible_work_groups;
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = [program_ maxTotalThreadsPerThreadgroup];
  kernel_info.private_memory_size = 0;
  operation_->GetPossibleKernelWorkGroups(tuning_type, device->GetInfo(),
                                          kernel_info, &possible_work_groups);
  if (possible_work_groups.empty()) {
    return absl::NotFoundError(
        "Can not found work_group size to launch kernel");
  }
  operation_->work_group_size_ = possible_work_groups[0];
  operation_->work_groups_count_ = GetWorkGroupsCount(
      operation_->grid_dimension_, operation_->grid_size_,
      operation_->work_group_size_, operation_->work_group_launch_order_);
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
