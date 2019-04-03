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

#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"

#include <map>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::metal::RuntimeOptions;
using ::tflite::gpu::InternalError;
using ::tflite::gpu::OkStatus;
using ::tflite::gpu::Status;
using ::tflite::gpu::ValueId;
using ::tflite::gpu::AlignByN;
using ::tflite::gpu::HalfBits;
using ::tflite::gpu::MemoryStrategy;
using ::tflite::gpu::TensorUsageRecord;

@implementation TFLInferenceContext {
  std::vector<TFLComputeTask*> _computeTasks;
  std::vector<ValueId> _outputIds;
  id<MTLDevice> _device;
  RuntimeOptions _options;
}

- (Status)compileModelWithDevice:(id<MTLDevice>)device
                 taskDescriptors:(const std::vector<ComputeTaskDescriptorPtr>&)taskDescriptors
                 outputBufferIDs:(const std::vector<ValueId>&)requestedOutputBufferIDs
                  runtimeOptions:(const RuntimeOptions&)options {
  _device = device;
  _outputIds = requestedOutputBufferIDs;
  _options = options;
  // Metal resources are created here.
  for (const auto& node : taskDescriptors) {
    TFLComputeTask* task = [[TFLComputeTask alloc] init];
    RETURN_IF_ERROR([task compileWithDevice:_device taskDescriptor:node runtimeOptions:_options]);
    _computeTasks.emplace_back(task);
  }
  return OkStatus();
}

- (Status)setInputDimensions:(const std::map<ValueId, BHWC>&)inputDimensions
            outputDimensions:(std::map<ValueId, BHWC>*)outputDimensions
             taskDescriptors:(const std::vector<ComputeTaskDescriptorPtr>&)taskDescriptors {
  // These maps contain all input/output/intermediate buffers shared across model.
  std::map<ValueId, BHWC> dimensions = inputDimensions;
  std::map<ValueId, id<MTLBuffer>> buffers;
  std::set<ValueId> preallocatedIds;
  // Insert uninitialized input buffers. This buffers will be set externally.
  for (auto dimension : dimensions) {
    buffers[dimension.first] = nil;
    preallocatedIds.insert(dimension.first);
  }
  for (const auto& outputId : _outputIds) {
    preallocatedIds.insert(outputId);
  }
  for (auto& task : _computeTasks) {
    // The same device must be used here as well as on shader compilation stage.
    RETURN_IF_ERROR([task setInputDimensionsWithDevice:_device dimensions:&dimensions]);
  }
  for (auto id : _outputIds) {
    (*outputDimensions)[id] = dimensions[id];
  }

  // TODO(ypisarchyk): it make sense to move it to separate function
  // Generate usage records for each intermediate tensor in order of their first_task
  std::vector<TensorUsageRecord> usageRecords;
  std::map<ValueId, size_t> usageRecordIds;
  for (uint32_t i = 0; i < taskDescriptors.size(); ++i) {
    auto outputId = taskDescriptors[i]->output_buffer.id;
    if (!preallocatedIds.count(outputId)) {
      if (!usageRecordIds.count(outputId)) {
        const auto it = dimensions.find(outputId);
        if (it == dimensions.end()) {
          return InternalError("Dimensions for intermediate tensor not found.");
        }
        usageRecordIds[outputId] = usageRecords.size();
        usageRecords.emplace_back(it->second.w * it->second.h * AlignByN(it->second.c, 4), i, i);
      } else {
        usageRecords[usageRecordIds[outputId]].last_task = i;
      }
    }
    for (auto& buffer : taskDescriptors[i]->input_buffers) {
      if (!preallocatedIds.count(buffer.id)) {
        usageRecords[usageRecordIds[buffer.id]].last_task = i;
      }
    }
  }

  tflite::gpu::ObjectsAssignment assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(usageRecords, MemoryStrategy::GREEDY, &assignment));
  auto objectsCount = assignment.object_sizes.size();
  std::vector<id<MTLBuffer>> sharedBuffers(objectsCount);
  size_t dataTypeSize = _options.storage_precision == RuntimeOptions::Precision::FP32
                            ? sizeof(float)
                            : sizeof(HalfBits);

  // allocate buffers for each shared object
  for (size_t i = 0; i < objectsCount; ++i) {
    // Initialize metal buffer
    NSUInteger bufferSize = dataTypeSize * assignment.object_sizes[i];

#if (defined(__MAC_10_14) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_14) ||      \
    (defined(__IPHONE_12_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_12_0) || \
    (defined(__TVOS_12_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_12_0)
    if (bufferSize > [_device maxBufferLength]) {
      std::string error("Tensor id: ");
      error += std::to_string(assignment.object_ids[i]) +
               " with size: " + std::to_string(bufferSize) +
               " exceeds MTLDevice maxBufferLength: " + std::to_string([_device maxBufferLength]);
      return ::tflite::gpu::ResourceExhaustedError(error);
    }
#endif
#if defined(__MAC_10_12) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_12
    if ([_device currentAllocatedSize] + bufferSize > [_device recommendedMaxWorkingSetSize]) {
      std::string error("Out of memory in MTLBuffer allocation. Currently allocated: ");
      error += std::to_string([_device currentAllocatedSize]);
      return ::tflite::gpu::ResourceExhaustedError(error);
    }
#endif

    sharedBuffers[i] = [_device newBufferWithLength:bufferSize
                                            options:MTLResourceStorageModeShared];
  }
  for (auto& task : _computeTasks) {
    RETURN_IF_ERROR([task assignBuffers:&buffers
                              outputIds:_outputIds
                         usageRecordIds:usageRecordIds
                        sharedBufferIds:assignment.object_ids
                          sharedBuffers:sharedBuffers]);
  }
  return OkStatus();
}

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
       inputOutputBuffers:(const std::map<ValueId, id<MTLBuffer>>&)inputOutputBuffers
             encoderBlock:(id<MTLComputeCommandEncoder> (^)(bool isLast))encoderBlock {
  for (int i = 0; i < _computeTasks.size(); ++i) {
    auto& task = _computeTasks[i];
    [task encodeWithEncoder:commandEncoder inputOutputBuffers:inputOutputBuffers];
    if (encoderBlock != nil) {
      commandEncoder = encoderBlock(i == _computeTasks.size() - 1);
    }
  }
}

@end
