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

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::AlignByN;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::InternalError;
using ::tflite::gpu::InvalidArgumentError;
using ::tflite::gpu::HalfBits;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::metal::CreateComputeProgram;
using ::tflite::gpu::metal::DispatchParamsFunction;
using ::tflite::gpu::metal::OutputDimensions;
using ::tflite::gpu::metal::RuntimeOptions;
using ::tflite::gpu::metal::UniformsFunction;
using ::tflite::gpu::OkStatus;
using ::tflite::gpu::Status;
using ::tflite::gpu::uint3;
using ::tflite::gpu::ValueId;

@implementation TFLComputeTask {
  struct InputBuffer {
    ValueId uid;
    id<MTLBuffer> metalHandle;
  };
  struct OutputBuffer {
    ValueId uid;
    id<MTLBuffer> metalHandle;
    OutputDimensions dimensionsFunction;
    std::vector<ValueId> alias;
  };
  struct UniformBuffer {
    std::vector<uint8_t> data;
    UniformsFunction dataFunction;
  };

  id<MTLComputePipelineState> _program;
  std::vector<InputBuffer> _inputBuffers;
  std::vector<OutputBuffer> _outputBuffers;
  std::vector<id<MTLBuffer>> _immutableBuffers;
  std::vector<UniformBuffer> _uniformBuffers;
  uint3 _groupsSize;
  uint3 _groupsCount;
  DispatchParamsFunction _resizeFunction;
  std::string _description;
}

- (Status)compileWithDevice:(id<MTLDevice>)device
             taskDescriptor:(ComputeTaskDescriptorPtr)desc
             runtimeOptions:(const RuntimeOptions&)options {
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
  if (options.storage_precision == RuntimeOptions::Precision::FP32) {
    storageType = @"float";
    accumulatorType = @"float";
  } else {
    // FP16
    storageType = @"half";
    if (options.accumulator_precision == RuntimeOptions::Precision::FP32) {
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
    @"BARRIER" : barrier,
  };

  NSString* code = [NSString stringWithCString:desc->shader_source.c_str()
                                      encoding:[NSString defaultCStringEncoding]];
  id<MTLComputePipelineState> program;
  RETURN_IF_ERROR(CreateComputeProgram(device, code, @"ComputeFunction", macros, &program));
  if (!program) {
    return InternalError("Unknown shader compilation error");
  }
  for (auto& buffer : desc->input_buffers) {
    _inputBuffers.emplace_back(InputBuffer{buffer.id, nil});
  }
  for (auto& uniform : desc->uniform_buffers) {
    _uniformBuffers.emplace_back(UniformBuffer{{}, uniform.data_function});
  }
  _outputBuffers.emplace_back(OutputBuffer{desc->output_buffer.id, nil,
                                           desc->output_buffer.dimensions_function,
                                           desc->output_buffer.alias});
  for (auto& immutable : desc->immutable_buffers) {
    int padding =
        4 * (options.storage_precision == RuntimeOptions::Precision::FP32 ? sizeof(float)
                                                                          : sizeof(HalfBits));
    int paddedSize = AlignByN(immutable.data.size(), padding);
    immutable.data.resize(paddedSize);
    id<MTLBuffer> metalBuffer = [device newBufferWithBytes:immutable.data.data()
                                                    length:immutable.data.size()
                                                   options:MTLResourceStorageModeShared];
    _immutableBuffers.emplace_back(metalBuffer);
  }
  _resizeFunction = desc->resize_function;
  _program = program;
  _description = desc->description;
  return OkStatus();
}

- (Status)setInputDimensionsWithDevice:(id<MTLDevice>)device
                            dimensions:
                                (std::map<::tflite::gpu::ValueId, ::tflite::gpu::BHWC>*)dimensions {
  // Re-calculate output buffers dimensions
  for (auto& buffer : _outputBuffers) {
    auto outputDimensions = buffer.dimensionsFunction(*dimensions);
    for (ValueId duplicate : buffer.alias) {
      (*dimensions)[duplicate] = outputDimensions;
    }
    // Store buffer dimensions
    (*dimensions)[buffer.uid] = outputDimensions;
  }

  for (auto& uniform : _uniformBuffers) {
    uniform.data = uniform.dataFunction(*dimensions);
  }

  // Dispatch parameters re-calculation
  auto workGroups = _resizeFunction(*dimensions);
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
    return InvalidArgumentError(error);
  }
  _groupsCount = workGroups.second;
  return OkStatus();
}

- (Status)assignBuffers:(std::map<::tflite::gpu::ValueId, id<MTLBuffer>>*)buffers
              outputIds:(const std::vector<::tflite::gpu::ValueId>&)outputIds
         usageRecordIds:(const std::map<ValueId, size_t>&)usageRecordIds
        sharedBufferIds:(const std::vector<size_t>&)sharedBufferIds
          sharedBuffers:(const std::vector<id<MTLBuffer>>&)sharedBuffers {
  for (auto& buffer : _outputBuffers) {
    // If the buffer is intermediate: set its metalHandle from sharedBuffers
    if (std::find(outputIds.begin(), outputIds.end(), buffer.uid) == outputIds.end()) {
      auto usageRecordIt = usageRecordIds.find(buffer.uid);
      if (usageRecordIt == usageRecordIds.end()) {
        return InternalError("TensorUsageRecord for intermediate tensor is not found.");
      }
      buffer.metalHandle = sharedBuffers.at(sharedBufferIds.at(usageRecordIt->second));
      (*buffers)[buffer.uid] = buffer.metalHandle;
    }
  }

  // Re-assign input buffers
  for (auto& buffer : _inputBuffers) {
    buffer.metalHandle = (*buffers)[buffer.uid];
  }
  return OkStatus();
}

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)encoder
       inputOutputBuffers:(const std::map<ValueId, id<MTLBuffer>>&)inputOutputBuffers {
  // The dispatch call is intended to be skipped.
  if (_groupsCount.x * _groupsCount.y * _groupsCount.z == 0) {
    return;
  }

  [encoder setComputePipelineState:_program];

  int bindIndex = 0;
  for (auto& buffer : _outputBuffers) {
    const auto externalBuffer = inputOutputBuffers.find(buffer.uid);
    if (externalBuffer == inputOutputBuffers.end()) {
      [encoder setBuffer:buffer.metalHandle offset:0 atIndex:bindIndex];
    } else {
      // the buffer is input or output
      [encoder setBuffer:externalBuffer->second offset:0 atIndex:bindIndex];
    }
    bindIndex++;
  }
  for (auto& buffer : _inputBuffers) {
    const auto externalBuffer = inputOutputBuffers.find(buffer.uid);
    if (externalBuffer == inputOutputBuffers.end()) {
      [encoder setBuffer:buffer.metalHandle offset:0 atIndex:bindIndex];
    } else {
      // the buffer is input or output
      [encoder setBuffer:externalBuffer->second offset:0 atIndex:bindIndex];
    }
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

  MTLSize groupsCount = MTLSizeMake(_groupsCount.x, _groupsCount.y, _groupsCount.z);
  MTLSize groupsSize = MTLSizeMake(_groupsSize.x, _groupsSize.y, _groupsSize.z);
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

@end
