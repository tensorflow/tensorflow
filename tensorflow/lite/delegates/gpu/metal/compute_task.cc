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
#include <string>
#include <tuple>

#include "tensorflow/lite/delegates/gpu/metal/metal_arguments.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {

absl::Status ComputeTask::CompileWithDevice(id<MTLDevice> device,
                                            const NodeDescriptor& desc,
                                            CalculationsPrecision precision) {
  size_t offset = desc.task->src_tensors_names.size() +
                  desc.task->uniform_buffers.size() +
                  desc.task->immutable_buffers.size() + 1;
  RETURN_IF_ERROR(metal_args_.Init(device, offset, &desc.task->args,
                                   &desc.task->shader_source));
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
      [NSString stringWithCString:desc.task->shader_source.c_str()
                         encoding:[NSString defaultCStringEncoding]];
  id<MTLComputePipelineState> program;
  RETURN_IF_ERROR(
      CreateComputeProgram(device, code, @"ComputeFunction", macros, &program));
  if (!program) {
    return absl::InternalError("Unknown shader compilation error");
  }
  for (auto& id : desc.src_tensors_ids) {
    input_buffers_.emplace_back(InputBuffer{id, nil});
  }
  for (auto& uniform : desc.task->uniform_buffers) {
    uniform_buffers_.emplace_back(UniformBuffer{{}, uniform.data_function});
  }
  output_buffers_.emplace_back(OutputBuffer{desc.dst_tensors_ids[0], nil});
  const bool f32_storage = precision == CalculationsPrecision::F32;
  for (auto& immutable : desc.task->immutable_buffers) {
    int padding = 4 * (f32_storage ? sizeof(float) : sizeof(HalfBits));
    int paddedSize = AlignByN(immutable.data.size(), padding);
    immutable.data.resize(paddedSize);
    id<MTLBuffer> metalBuffer =
        [device newBufferWithBytes:immutable.data.data()
                            length:immutable.data.size()
                           options:MTLResourceStorageModeShared];
    immutable_buffers_.emplace_back(metalBuffer);
  }
  resize_function_ = desc.task->resize_function;
  program_ = program;
  src_tensors_names_ = desc.task->src_tensors_names;
  dst_tensors_names_ = desc.task->dst_tensors_names;
  tensors_as_args_ = desc.task->tensors_as_args;
  return absl::OkStatus();
}

absl::Status ComputeTask::UpdateParamsWithDevice(
      id<MTLDevice> device, const std::map<ValueId, BHWC>& tensor_shapes) {
  std::vector<BHWC> src_shapes;
  std::vector<BHWC> dst_shapes;
  for (const auto& in_buf : input_buffers_) {
    auto it = tensor_shapes.find(in_buf.uid);
    if (it == tensor_shapes.end()) {
      return absl::InvalidArgumentError("Missing tensor shape");
    }
    src_shapes.push_back(it->second);
  }
  for (const auto& out_buf : output_buffers_) {
    auto it = tensor_shapes.find(out_buf.uid);
    if (it == tensor_shapes.end()) {
      return absl::InvalidArgumentError("Missing tensor shape");
    }
    dst_shapes.push_back(it->second);
  }
  for (auto& uniform : uniform_buffers_) {
    uniform.data = uniform.data_function(src_shapes, dst_shapes);
  }

  // Dispatch parameters re-calculation
  auto workGroups = resize_function_(src_shapes, dst_shapes);
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
    if (ids.count(buffer.uid)) {
      return true;
    }
  }
  for (auto& buffer : output_buffers_) {
    if (ids.count(buffer.uid)) {
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
  for (const auto& buffer : output_buffers_) {
    [encoder setBuffer:buffer.metal_handle offset:0 atIndex:bindIndex];
    bindIndex++;
  }
  for (const auto& buffer : input_buffers_) {
    [encoder setBuffer:buffer.metal_handle offset:0 atIndex:bindIndex];
    bindIndex++;
  }
  for (auto& immutable : immutable_buffers_) {
    [encoder setBuffer:immutable offset:0 atIndex:bindIndex];
    bindIndex++;
  }
  for (auto& uniform : uniform_buffers_) {
    [encoder setBytes:uniform.data.data()
               length:uniform.data.size()
              atIndex:bindIndex];
    bindIndex++;
  }
  metal_args_.Encode(encoder, bindIndex);

  MTLSize groupsCount =
      MTLSizeMake(groups_count_.x, groups_count_.y, groups_count_.z);
  MTLSize groupsSize =
      MTLSizeMake(groups_size_.x, groups_size_.y, groups_size_.z);
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

std::vector<ValueId> ComputeTask::GetOutputIds() const {
  std::vector<tflite::gpu::ValueId> result;
  for (auto& buffer : output_buffers_) {
    result.push_back(buffer.uid);
  }
  return result;
}

std::vector<ValueId> ComputeTask::GetInputIds() const {
  std::vector<tflite::gpu::ValueId> result;
  for (auto& buffer : input_buffers_) {
    result.push_back(buffer.uid);
  }
  return result;
}

void ComputeTask::SetSrcTensor(const MetalSpatialTensor& tensor, int index) {
  input_buffers_[index].metal_handle = tensor.GetBufferHandle();
  if (tensors_as_args_) {
    auto name = src_tensors_names_[index];
    // extracting tensor_name from "device FLT4* tensor_name_buffer";
    name = name.substr(13, name.size() - 20);
    auto status = metal_args_.SetObjectRef(name, tensor);
  }
}

void ComputeTask::SetDstTensor(const MetalSpatialTensor& tensor, int index) {
  output_buffers_[index].metal_handle = tensor.GetBufferHandle();
  if (tensors_as_args_) {
    auto name = dst_tensors_names_[index];
    // extracting tensor_name from "device FLT4* tensor_name_buffer";
    name = name.substr(13, name.size() - 20);
    auto status = metal_args_.SetObjectRef(name, tensor);
  }
}

void ComputeTask::SetDescription(const std::string& description) {
  description_ = description;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
