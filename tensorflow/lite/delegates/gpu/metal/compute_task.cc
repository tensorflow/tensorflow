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

ComputeTask::ComputeTask(ComputeTask&& task)
    : operation_(std::move(task.operation_)),
      program_(task.program_),
      metal_args_(std::move(task.metal_args_)),
      use_arguments_buffer_(task.use_arguments_buffer_),
      need_icb_support_(task.need_icb_support_),
      arguments_encoder_(task.arguments_encoder_),
      arg_buffer_(task.arg_buffer_) {
  task.program_ = nullptr;
  task.arguments_encoder_ = nullptr;
  task.arg_buffer_ = nullptr;
}

ComputeTask& ComputeTask::operator=(ComputeTask&& task) {
  if (this != &task) {
    Release();
    operation_ = std::move(task.operation_);
    std::swap(program_, task.program_);
    metal_args_ = std::move(task.metal_args_);
    std::swap(use_arguments_buffer_, task.use_arguments_buffer_);
    std::swap(need_icb_support_, task.need_icb_support_);
    std::swap(arguments_encoder_, task.arguments_encoder_);
    std::swap(arg_buffer_, task.arg_buffer_);
  }
  return *this;
}

ComputeTask::~ComputeTask() { Release(); }

void ComputeTask::Release() {
  if (program_) {
    program_ = nullptr;
  }
  if (arguments_encoder_) {
    arguments_encoder_ = nullptr;
  }
  if (arg_buffer_) {
    arg_buffer_ = nullptr;
  }
}

void ComputeTask::Init(std::unique_ptr<GPUOperation>&& operation) {
  operation_ = std::move(operation);
}

const OperationDef& ComputeTask::GetDefinition() const {
  return operation_->GetDefinition();
}

bool ComputeTask::IsLinkable() const { return operation_->IsLinkable(); }

absl::Status ComputeTask::AddTask(ComputeTask* task) {
  return operation_->AddOperation(task->operation_.get());
}

absl::Status ComputeTask::Compile(MetalDevice* device) {
  RETURN_IF_ERROR(metal_args_.Init(use_arguments_buffer_, device,
                                   &operation_->args_, &operation_->code_));

  operation_->args_.ReleaseCPURepresentation();

  return CompileProgram(device, operation_->GetDefinition().precision,
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
    @"\"SELECT_BY_INDEX_FROM_FLT4(value, index)\"" : @"\"(value)[index]\"",
  };

  NSString* code =
      [NSString stringWithCString:kernel_code.c_str()
                         encoding:[NSString defaultCStringEncoding]];
  if (use_arguments_buffer_) {
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
      id<MTLFunction> function;
      RETURN_IF_ERROR(CreateFunction(device->device(), code, @"ComputeFunction",
                                     macros, &function));
      arguments_encoder_ = [function newArgumentEncoderWithBufferIndex:0];
      if (!arguments_encoder_) {
        return absl::InternalError("Failed to get MTLArgumentEncoder.");
      }
      arg_buffer_ =
          [device->device() newBufferWithLength:arguments_encoder_.encodedLength
                                        options:0];
      if (!arg_buffer_) {
        return absl::InternalError("Failed to create MTLBuffer.");
      }
      MTLComputePipelineDescriptor* pipeline_desc =
          [[MTLComputePipelineDescriptor alloc] init];
      pipeline_desc.computeFunction = function;
      if (need_icb_support_) {
        if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
          pipeline_desc.supportIndirectCommandBuffers = TRUE;
        } else {
          return absl::InternalError(
              "Indirect compute command buffer available since ios 13");
        }
      }
      NSError* error = nil;
      program_ = [device->device()
          newComputePipelineStateWithDescriptor:pipeline_desc
                                        options:MTLPipelineOptionNone
                                     reflection:nullptr
                                          error:&error];
      if (!program_) {
        NSString* error_string = [NSString
            stringWithFormat:@"newComputePipelineStateWithDescriptor: %@",
                             [error localizedDescription]];
        return absl::InternalError([error_string UTF8String]);
      }
    } else {
      return absl::InternalError(
          "Metal argument buffers available since ios 11.");
    }
  } else {
    id<MTLComputePipelineState> program;
    RETURN_IF_ERROR(CreateComputeProgram(device->device(), code,
                                         @"ComputeFunction", macros, &program));
    if (!program) {
      return absl::InternalError("Unknown shader compilation error");
    }
    program_ = program;
  }
  return absl::OkStatus();
}

absl::Status ComputeTask::UpdateParams() {
  for (int i = 0; i < operation_->GetSrcTensorsNames().size(); ++i) {
    const auto* metal_spatial_tensor =
        dynamic_cast<const MetalSpatialTensor*>(operation_->GetSrcTensors()[i]);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    RETURN_IF_ERROR(metal_args_.SetObjectRef(
        operation_->GetSrcTensorsNames()[i], *metal_spatial_tensor));
  }
  for (int i = 0; i < operation_->GetDstTensorsNames().size(); ++i) {
    const auto* metal_spatial_tensor =
        dynamic_cast<const MetalSpatialTensor*>(operation_->GetDstTensors()[i]);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    RETURN_IF_ERROR(metal_args_.SetObjectRef(
        operation_->GetDstTensorsNames()[i], *metal_spatial_tensor));
  }
  RETURN_IF_ERROR(operation_->BindArguments(&metal_args_));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  Update();
  return absl::OkStatus();
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void ComputeTask::EncodeToICB(id<MTLIndirectComputeCommand> icb_command) {
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->GetWorkGroupsCount().x;
  groupsCount.height = operation_->GetWorkGroupsCount().y;
  groupsCount.depth = operation_->GetWorkGroupsCount().z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [icb_command setComputePipelineState:program_];
  [icb_command setKernelBuffer:arg_buffer_ offset:0 atIndex:0];
  [icb_command concurrentDispatchThreadgroups:groupsCount
                        threadsPerThreadgroup:groupsSize];
  [icb_command setBarrier];
}

API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
void ComputeTask::AddResourcesToEncoder(
    id<MTLComputeCommandEncoder> encoder) const {
  metal_args_.AddResourcesToEncoder(encoder);
}

void ComputeTask::Update() {
  if (use_arguments_buffer_) {
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
      [arguments_encoder_ setArgumentBuffer:arg_buffer_ offset:0];
      metal_args_.EncodeArguments(arguments_encoder_);
    }
  }
}

void ComputeTask::Encode(id<MTLComputeCommandEncoder> encoder) {
  [encoder setComputePipelineState:program_];
  if (use_arguments_buffer_) {
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
      metal_args_.AddResourcesToEncoder(encoder);
      [encoder setBuffer:arg_buffer_ offset:0 atIndex:0];
    }
  } else {
    metal_args_.Encode(encoder, 0);
  }
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->GetWorkGroupsCount().x;
  groupsCount.height = operation_->GetWorkGroupsCount().y;
  groupsCount.depth = operation_->GetWorkGroupsCount().z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

void ComputeTask::SetSrcTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetSrc(tensor, index);
  auto status = metal_args_.SetObjectRef(
      operation_->GetSrcTensorsNames()[index], *tensor);
}

void ComputeTask::SetDstTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetDst(tensor, index);
  auto status = metal_args_.SetObjectRef(
      operation_->GetDstTensorsNames()[index], *tensor);
}

absl::Status ComputeTask::Tune(TuningType tuning_type, MetalDevice* device) {
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = [program_ maxTotalThreadsPerThreadgroup];
  kernel_info.private_memory_size = 0;
  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(tuning_type, device->GetInfo(), kernel_info,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch kernel");
  }
  operation_->work_group_size_ = possible_dispatches[0].work_group_size;
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
