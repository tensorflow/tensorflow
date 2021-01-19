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
#include "tensorflow/lite/delegates/gpu/common/model.h"
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

void ComputeTask::Init(std::unique_ptr<ComputeTaskDescriptor>&& task_desc) {
  task_desc_ = std::move(task_desc);
}

void ComputeTask::Init(std::unique_ptr<GPUOperation>&& operation) {
  operation_ = std::move(operation);
}

const OperationDef& ComputeTask::GetDefinition() const {
  if (task_desc_) {
    return task_desc_->definition;
  } else {
    return operation_->definition_;
  }
}

bool ComputeTask::IsLinkable() const {
  if (task_desc_) {
    return task_desc_->is_linkable;
  } else {
    return operation_->IsLinkable();
  }
}

absl::Status ComputeTask::AddTask(ComputeTask* task) {
  if (task_desc_ && task->task_desc_) {
    return task_desc_->AddTask(task->task_desc_.get());
  }
  if (task_desc_ && task->operation_) {
    return task_desc_->AddOperation(task->operation_.get());
  }
  if (operation_ && task->operation_) {
    return operation_->AddOperation(task->operation_.get());
  }
  return absl::UnimplementedError(
      "Not implemented this combination of task fusion");
}

absl::Status ComputeTask::Compile(MetalDevice* device) {
  if (task_desc_) {
    return CompileTask(device);
  } else {
    return CompileOperation(device);
  }
}

absl::Status ComputeTask::CompileTask(MetalDevice* device) {
  task_desc_->AssembleCode();
  const std::map<std::string, std::string> linkables = {
      {task_desc_->dst_tensors_names[0], task_desc_->elementwise_code}};
  RETURN_IF_ERROR(metal_args_.Init(linkables, device, &task_desc_->args,
                                   &task_desc_->shader_source));
  task_desc_->args.ReleaseCPURepresentation();

  return CompileProgram(device, task_desc_->definition.precision,
                        task_desc_->shader_source);
}

absl::Status ComputeTask::CompileOperation(MetalDevice* device) {
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
  NSString* toAccumulatorType = @"";
  NSString* toAccumulatorType2 = @"";
  NSString* toAccumulatorType3 = @"";
  NSString* toAccumulatorType4 = @"";
  if (precision == CalculationsPrecision::F32) {
    storageType = @"float";
    accumulatorType = @"float";
  } else {
    // FP16
    storageType = @"half";
    if (precision == CalculationsPrecision::F32_F16) {
      accumulatorType = @"float";
      toAccumulatorType = @"float";
      toAccumulatorType2 = @"float2";
      toAccumulatorType3 = @"float3";
      toAccumulatorType4 = @"float4";
    } else {
      accumulatorType = @"half";
    }
  }
  NSDictionary<NSString*, NSString*>* macros = @{
    @"FLT" : storageType,
    @"FLT2" : [NSString stringWithFormat:@"%@2", storageType],
    @"FLT3" : [NSString stringWithFormat:@"%@3", storageType],
    @"FLT4" : [NSString stringWithFormat:@"%@4", storageType],
    @"ACCUM_FLT" : accumulatorType,
    @"ACCUM_FLT2" : [NSString stringWithFormat:@"%@2", accumulatorType],
    @"ACCUM_FLT3" : [NSString stringWithFormat:@"%@3", accumulatorType],
    @"ACCUM_FLT4" : [NSString stringWithFormat:@"%@4", accumulatorType],
    @"TO_ACCUM_TYPE" : toAccumulatorType,
    @"TO_ACCUM2_TYPE" : toAccumulatorType2,
    @"TO_ACCUM3_TYPE" : toAccumulatorType3,
    @"TO_ACCUM4_TYPE" : toAccumulatorType4,
    @"SIMDGROUP_BARRIER" : barrier,
    @"MAIN_FUNCTION" : @"\"kernel void ComputeFunction\"",
    @"GLOBAL_ID_0" : @"static_cast<int>(reserved_gid.x)",
    @"GLOBAL_ID_1" : @"static_cast<int>(reserved_gid.y)",
    @"GLOBAL_ID_2" : @"static_cast<int>(reserved_gid.z)",
    @"INIT_FLT(value)" : [NSString stringWithFormat:@"%@(value)", storageType],
    @"INIT_FLT4(value)" :
        [NSString stringWithFormat:@"%@4(value)", storageType],
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

absl::Status ComputeTask::UpdateParams(const GpuInfo& gpu_info,
                                       const std::vector<BHWC>& src_shapes,
                                       const std::vector<BHWC>& dst_shapes) {
  if (task_desc_) {
    return UpdateTaskParams(gpu_info, src_shapes, dst_shapes);
  } else {
    return UpdateOperationParams();
  }
}

absl::Status ComputeTask::UpdateTaskParams(
    const GpuInfo& gpu_info, const std::vector<BHWC>& src_shapes,
    const std::vector<BHWC>& dst_shapes) {
  RETURN_IF_ERROR(
      task_desc_->update_function(src_shapes, dst_shapes, &metal_args_));

  // Dispatch parameters re-calculation
  auto workGroups = task_desc_->resize_function(src_shapes, dst_shapes);
  groups_size_ = workGroups.first;
  if (groups_size_.x > gpu_info.GetMaxWorkGroupSizeForX() ||
      groups_size_.y > gpu_info.GetMaxWorkGroupSizeForY() ||
      groups_size_.z > gpu_info.GetMaxWorkGroupSizeForZ()) {
    std::string error("Threads per working group: ");
    error += std::to_string(groups_size_.x) + ", " +
             std::to_string(groups_size_.y) + ", " +
             std::to_string(groups_size_.z);
    error += "is larger than the MTLDevice can support: ";
    error += std::to_string(gpu_info.GetMaxWorkGroupSizeForX()) + ", " +
             std::to_string(gpu_info.GetMaxWorkGroupSizeForY()) + ", " +
             std::to_string(gpu_info.GetMaxWorkGroupSizeForZ());
    return absl::InvalidArgumentError(error);
  }
  groups_count_ = workGroups.second;
  return absl::OkStatus();
}

absl::Status ComputeTask::UpdateOperationParams() {
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
  if (task_desc_) {
    return EncodeTask(encoder);
  } else {
    return EncodeOperation(encoder);
  }
}

void ComputeTask::EncodeTask(id<MTLComputeCommandEncoder> encoder) {
  // The dispatch call is intended to be skipped.
  if (groups_count_.x * groups_count_.y * groups_count_.z == 0) {
    return;
  }

  [encoder setComputePipelineState:program_];

  metal_args_.Encode(encoder, 0);

  MTLSize groupsCount =
      MTLSizeMake(groups_count_.x, groups_count_.y, groups_count_.z);
  MTLSize groupsSize =
      MTLSizeMake(groups_size_.x, groups_size_.y, groups_size_.z);
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

void ComputeTask::EncodeOperation(id<MTLComputeCommandEncoder> encoder) {
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
  if (task_desc_) {
    auto status =
        metal_args_.SetObjectRef(task_desc_->src_tensors_names[index], *tensor);
  } else {
    operation_->SetSrc(tensor, index);
    auto status = metal_args_.SetObjectRef(
        operation_->src_tensors_names_[index], *tensor);
  }
}

void ComputeTask::SetDstTensor(MetalSpatialTensor* tensor, int index) {
  if (task_desc_) {
    auto status =
        metal_args_.SetObjectRef(task_desc_->dst_tensors_names[index], *tensor);
  } else {
    operation_->SetDst(tensor, index);
    auto status = metal_args_.SetObjectRef(
        operation_->dst_tensors_names_[index], *tensor);
  }
}

absl::Status ComputeTask::Tune(TuningType tuning_type, MetalDevice* device) {
  if (!operation_) {
    // Tune supported only in GPUOperation
    return absl::OkStatus();
  }
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
