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

using ::tflite::gpu::AlignByN;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::HalfBits;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::metal::CreateComputeProgram;
using ::tflite::gpu::metal::DispatchParamsFunction;
using ::tflite::gpu::CalculationsPrecision;
using ::tflite::gpu::metal::UniformsFunction;
using ::tflite::gpu::uint3;
using ::tflite::gpu::ValueId;

namespace {

struct InputBuffer {
  ValueId uid;
  id<MTLBuffer> metalHandle;
};

struct OutputBuffer {
  ValueId uid;
  id<MTLBuffer> metalHandle;
};

struct UniformBuffer {
  std::vector<uint8_t> data;
  UniformsFunction dataFunction;
};

}  // namespace

@implementation TFLComputeTask {
  id<MTLComputePipelineState> _program;
  std::vector<InputBuffer> _inputBuffers;
  std::vector<OutputBuffer> _outputBuffers;
  std::vector<id<MTLBuffer>> _immutableBuffers;
  std::vector<UniformBuffer> _uniformBuffers;
  uint3 _groupsSize;
  uint3 _groupsCount;
  DispatchParamsFunction _resizeFunction;
  std::string _description;
  tflite::gpu::metal::MetalArguments _metal_args;
}

- (absl::Status)compileWithDevice:(id<MTLDevice>)device
                   taskDescriptor:(const tflite::gpu::metal::NodeDescriptor&)desc
                        precision:(CalculationsPrecision)precision; {
  size_t offset = desc.task->src_tensors_names.size() + desc.task->uniform_buffers.size()
                  + desc.task->immutable_buffers.size() + 1;
  RETURN_IF_ERROR(_metal_args.Init(device, offset, &desc.task->args, &desc.task->shader_source));
  NSString* barrier;
  // simdgroup_barrier is supported on macOS 10.13+ and Metal shading language version 2.0
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

  NSString* code = [NSString stringWithCString:desc.task->shader_source.c_str()
                                      encoding:[NSString defaultCStringEncoding]];
  id<MTLComputePipelineState> program;
  RETURN_IF_ERROR(CreateComputeProgram(device, code, @"ComputeFunction", macros, &program));
  if (!program) {
    return absl::InternalError("Unknown shader compilation error");
  }
  for (auto& id : desc.src_tensors_ids) {
    _inputBuffers.emplace_back(InputBuffer{id, nil});
  }
  for (auto& uniform : desc.task->uniform_buffers) {
    _uniformBuffers.emplace_back(UniformBuffer{{}, uniform.data_function});
  }
  _outputBuffers.emplace_back(OutputBuffer{desc.dst_tensors_ids[0], nil});
  const bool f32_storage = precision == CalculationsPrecision::F32;
  for (auto& immutable : desc.task->immutable_buffers) {
    int padding = 4 * (f32_storage ? sizeof(float) : sizeof(HalfBits));
    int paddedSize = AlignByN(immutable.data.size(), padding);
    immutable.data.resize(paddedSize);
    id<MTLBuffer> metalBuffer = [device newBufferWithBytes:immutable.data.data()
                                                    length:immutable.data.size()
                                                   options:MTLResourceStorageModeShared];
    _immutableBuffers.emplace_back(metalBuffer);
  }
  _resizeFunction = desc.task->resize_function;
  _program = program;
  return absl::OkStatus();
}

- (absl::Status)
    updateParamsWithDevice:(id<MTLDevice>)device
              tensorShapes:(const std::map<tflite::gpu::ValueId, tflite::gpu::BHWC>&)tensorShapes {
  std::vector<BHWC> src_shapes;
  std::vector<BHWC> dst_shapes;
  for (const auto& in_buf : _inputBuffers) {
    auto it = tensorShapes.find(in_buf.uid);
    if (it == tensorShapes.end()) {
      return absl::InvalidArgumentError("Missing tensor shape");
    }
    src_shapes.push_back(it->second);
  }
  for (const auto& out_buf : _outputBuffers) {
    auto it = tensorShapes.find(out_buf.uid);
    if (it == tensorShapes.end()) {
      return absl::InvalidArgumentError("Missing tensor shape");
    }
    dst_shapes.push_back(it->second);
  }
  for (auto& uniform : _uniformBuffers) {
    uniform.data = uniform.dataFunction(src_shapes, dst_shapes);
  }

  // Dispatch parameters re-calculation
  auto workGroups = _resizeFunction(src_shapes, dst_shapes);
  _groupsSize = workGroups.first;
  MTLSize threadsPerGroup = [device maxThreadsPerThreadgroup];
  if (_groupsSize.x > threadsPerGroup.width || _groupsSize.y > threadsPerGroup.height ||
      _groupsSize.z > threadsPerGroup.depth) {
    std::string error("Threads per working group: ");
    error += std::to_string(_groupsSize.x) + ", " + std::to_string(_groupsSize.y) + ", " +
             std::to_string(_groupsSize.z);
    error += "is larger than the MTLDevice can support: ";
    error += std::to_string(threadsPerGroup.width) + ", " + std::to_string(threadsPerGroup.height) +
             ", " + std::to_string(threadsPerGroup.depth);
    return absl::InvalidArgumentError(error);
  }
  _groupsCount = workGroups.second;
  return absl::OkStatus();
}

- (absl::Status)assignBuffers:(std::map<::tflite::gpu::ValueId, id<MTLBuffer>>*)buffers
                    outputIds:(const std::vector<::tflite::gpu::ValueId>&)outputIds
               usageRecordIds:(const std::map<ValueId, size_t>&)usageRecordIds
              sharedBufferIds:(const std::vector<size_t>&)sharedBufferIds
                sharedBuffers:(const std::vector<id<MTLBuffer>>&)sharedBuffers {
  for (auto& buffer : _outputBuffers) {
    // If the buffer is intermediate: set its metalHandle from sharedBuffers
    if (std::find(outputIds.begin(), outputIds.end(), buffer.uid) == outputIds.end()) {
      auto usageRecordIt = usageRecordIds.find(buffer.uid);
      if (usageRecordIt == usageRecordIds.end()) {
        return absl::InternalError("TensorUsageRecord for intermediate tensor is not found.");
      }
      buffer.metalHandle = sharedBuffers.at(sharedBufferIds.at(usageRecordIt->second));
      (*buffers)[buffer.uid] = buffer.metalHandle;
    }
  }

  // Re-assign input buffers
  for (auto& buffer : _inputBuffers) {
    buffer.metalHandle = (*buffers)[buffer.uid];
  }
  return absl::OkStatus();
}

- (bool)hasInOutIds:(const std::set<::tflite::gpu::ValueId>&)ids {
  for (auto& buffer : _inputBuffers) {
    if (ids.count(buffer.uid)) {
      return true;
    }
  }
  for (auto& buffer : _outputBuffers) {
    if (ids.count(buffer.uid)) {
      return true;
    }
  }
  return false;
}

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)encoder {
  // The dispatch call is intended to be skipped.
  if (_groupsCount.x * _groupsCount.y * _groupsCount.z == 0) {
    return;
  }

  [encoder setComputePipelineState:_program];

  int bindIndex = 0;
  for (const auto& buffer : _outputBuffers) {
    [encoder setBuffer:buffer.metalHandle offset:0 atIndex:bindIndex];
    bindIndex++;
  }
  for (const auto& buffer : _inputBuffers) {
    [encoder setBuffer:buffer.metalHandle offset:0 atIndex:bindIndex];
    bindIndex++;
  }
  for (auto& immutable : _immutableBuffers) {
    [encoder setBuffer:immutable offset:0 atIndex:bindIndex];
    bindIndex++;
  }
  for (auto& uniform : _uniformBuffers) {
    [encoder setBytes:uniform.data.data() length:uniform.data.size() atIndex:bindIndex];
    bindIndex++;
  }
  _metal_args.Encode(encoder, bindIndex);

  MTLSize groupsCount = MTLSizeMake(_groupsCount.x, _groupsCount.y, _groupsCount.z);
  MTLSize groupsSize = MTLSizeMake(_groupsSize.x, _groupsSize.y, _groupsSize.z);
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

- (std::vector<tflite::gpu::ValueId>)getOutputIds {
  std::vector<tflite::gpu::ValueId> result;
  for (auto& buffer : _outputBuffers) {
    result.push_back(buffer.uid);
  }
  return result;
}

- (std::vector<tflite::gpu::ValueId>)getInputIds {
  std::vector<tflite::gpu::ValueId> result;
  for (auto& buffer : _inputBuffers) {
    result.push_back(buffer.uid);
  }
  return result;
}

- (void)setSrcTensor:(const tflite::gpu::metal::MetalSpatialTensor&)tensor
           withIndex:(int)index; {
  _inputBuffers[index].metalHandle = tensor.GetBufferHandle();
}

- (void)setDstTensor:(const tflite::gpu::metal::MetalSpatialTensor&)tensor
           withIndex:(int)index; {
  _outputBuffers[index].metalHandle = tensor.GetBufferHandle();
}

- (void)setDescription:(const std::string&)description {
  _description = description;
}

@end
