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
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {

void ComputeTask::Init(std::unique_ptr<ComputeTaskDescriptor>&& task_desc,
                       const std::vector<ValueId>& input_ids,
                       const std::vector<ValueId>& output_ids) {
  task_desc_ = std::move(task_desc);
  input_buffers_ = input_ids;
  output_buffers_ = output_ids;
}

absl::Status ComputeTask::Compile(id<MTLDevice> device,
                                  CalculationsPrecision precision) {
  task_desc_->AssembleCode();
  const std::map<std::string, std::string> linkables = {
      {task_desc_->dst_tensors_names[0], task_desc_->elementwise_code}};
  RETURN_IF_ERROR(metal_args_.Init(device, linkables, &task_desc_->args,
                                   &task_desc_->shader_source));
  task_desc_->args.ReleaseCPURepresentation();
  NSString* barrier;
  // simdgroup_barrier is supported on macOS 10.13+ and Metal shading language
  // version 2.0
  if (@available(macOS 10.13, iOS 10.0, tvOS 10.0, *)) {
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
  };

  NSString* code =
      [NSString stringWithCString:task_desc_->shader_source.c_str()
                         encoding:[NSString defaultCStringEncoding]];
  id<MTLComputePipelineState> program;
  RETURN_IF_ERROR(
      CreateComputeProgram(device, code, @"ComputeFunction", macros, &program));
  if (!program) {
    return absl::InternalError("Unknown shader compilation error");
  }
  program_ = program;
  return absl::OkStatus();
}

absl::Status ComputeTask::UpdateParams(id<MTLDevice> device,
                                       const std::vector<BHWC>& src_shapes,
                                       const std::vector<BHWC>& dst_shapes) {
  RETURN_IF_ERROR(
      task_desc_->update_function(src_shapes, dst_shapes, &metal_args_));

  // Dispatch parameters re-calculation
  auto workGroups = task_desc_->resize_function(src_shapes, dst_shapes);
  groups_size_ = workGroups.first;
  MTLSize threadsPerGroup = [device maxThreadsPerThreadgroup];
  if (groups_size_.x > threadsPerGroup.width ||
      groups_size_.y > threadsPerGroup.height ||
      groups_size_.z > threadsPerGroup.depth) {
    std::string error("Threads per working group: ");
    error += std::to_string(groups_size_.x) + ", " +
             std::to_string(groups_size_.y) + ", " +
             std::to_string(groups_size_.z);
    error += "is larger than the MTLDevice can support: ";
    error += std::to_string(threadsPerGroup.width) + ", " +
             std::to_string(threadsPerGroup.height) + ", " +
             std::to_string(threadsPerGroup.depth);
    return absl::InvalidArgumentError(error);
  }
  groups_count_ = workGroups.second;
  return absl::OkStatus();
}

bool ComputeTask::HasInOutIds(const std::set<ValueId>& ids) const {
  for (auto& buffer : input_buffers_) {
    if (ids.count(buffer)) {
      return true;
    }
  }
  for (auto& buffer : output_buffers_) {
    if (ids.count(buffer)) {
      return true;
    }
  }
  return false;
}

void ComputeTask::EncodeWithEncoder(id<MTLComputeCommandEncoder> encoder) {
  // The dispatch call is intended to be skipped.
  if (groups_count_.x * groups_count_.y * groups_count_.z == 0) {
    return;
  }

  [encoder setComputePipelineState:program_];

  int bindIndex = 0;
  metal_args_.Encode(encoder, bindIndex);

  MTLSize groupsCount =
      MTLSizeMake(groups_count_.x, groups_count_.y, groups_count_.z);
  MTLSize groupsSize =
      MTLSizeMake(groups_size_.x, groups_size_.y, groups_size_.z);
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

std::vector<ValueId> ComputeTask::GetOutputIds() const {
  return output_buffers_;
}

std::vector<ValueId> ComputeTask::GetInputIds() const { return input_buffers_; }

void ComputeTask::SetSrcTensor(const MetalSpatialTensor& tensor, int index) {
  auto status =
      metal_args_.SetObjectRef(task_desc_->src_tensors_names[index], tensor);
}

void ComputeTask::SetDstTensor(const MetalSpatialTensor& tensor, int index) {
  auto status =
      metal_args_.SetObjectRef(task_desc_->dst_tensors_names[index], tensor);
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
