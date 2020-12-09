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
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::CalculationsPrecision;
using ::tflite::gpu::ValueId;
using ::tflite::gpu::AlignByN;
using ::tflite::gpu::HalfBits;
using ::tflite::gpu::MemoryStrategy;
using ::tflite::gpu::TensorUsageRecord;

@implementation TFLInferenceContext {
  std::vector<TFLComputeTask*> _computeTasks;
  // contains indexes of _computeTasks
  std::vector<int> _taskIdsWithInOutBuffers;
  std::vector<ValueId> _inputIds;
  std::vector<ValueId> _outputIds;
  id<MTLDevice> _device;
  CalculationsPrecision _precision;
  std::map<ValueId, BHWC> _tensorShapes;
}

- (absl::Status)compileModelWithDevice:(id<MTLDevice>)device
                                 model:(const tflite::gpu::metal::CompiledModel&) compiledModel
                        inputBufferIDs:(const std::vector<tflite::gpu::ValueId>&)inputBufferIDs
                       outputBufferIDs:(const std::vector<tflite::gpu::ValueId>&)outputBufferIDs
                             precision:(tflite::gpu::CalculationsPrecision)precision {
  _device = device;
  _inputIds = inputBufferIDs;
  _outputIds = outputBufferIDs;
  _precision = precision;
  // Metal resources are created here.
  for (const auto& node : compiledModel.nodes) {
    TFLComputeTask* task = [[TFLComputeTask alloc] init];
    RETURN_IF_ERROR([task compileWithDevice:_device
                             taskDescriptor:node
                                  precision:_precision]);
    [task setDescription:node.description];
    _computeTasks.emplace_back(task);
  }
  _tensorShapes = compiledModel.tensor_shapes;
  [self allocateTensors];
  return absl::OkStatus();
}

- (absl::Status)allocateTensors {
  // These maps contain all input/output/intermediate buffers shared across model.
  std::map<ValueId, id<MTLBuffer>> buffers;
  std::set<ValueId> preallocatedIds;
  // Insert uninitialized input buffers. This buffers will be set externally.
  for (auto tensor_id : _inputIds) {
    buffers[tensor_id] = nil;
    preallocatedIds.insert(tensor_id);
  }
  for (const auto& outputId : _outputIds) {
    preallocatedIds.insert(outputId);
  }
  for (auto& task : _computeTasks) {
    // The same device must be used here as well as on shader compilation stage.
    RETURN_IF_ERROR([task updateParamsWithDevice:_device tensorShapes:_tensorShapes]);
  }

  // TODO(ypisarchyk): it make sense to move it to separate function
  // Generate usage records for each intermediate tensor in order of their first_task
  std::vector<TensorUsageRecord<size_t>> usageRecords;
  std::map<ValueId, size_t> usageRecordIds;
  for (uint32_t i = 0; i < _computeTasks.size(); ++i) {
    for (const auto tensor_id : [_computeTasks[i] getOutputIds]) {
      if (!preallocatedIds.count(tensor_id)) {
        if (!usageRecordIds.count(tensor_id)) {
          const auto it = _tensorShapes.find(tensor_id);
          if (it == _tensorShapes.end()) {
            return absl::InternalError("Dimensions for intermediate tensor not found.");
          }
          usageRecordIds[tensor_id] = usageRecords.size();
          usageRecords.emplace_back(it->second.w * it->second.h * AlignByN(it->second.c, 4), i, i);
        } else {
          usageRecords[usageRecordIds[tensor_id]].last_task = i;
        }
      }
    }
    for (const auto tensor_id : [_computeTasks[i] getInputIds]) {
      if (!preallocatedIds.count(tensor_id)) {
        usageRecords[usageRecordIds[tensor_id]].last_task = i;
      }
    }
  }

  tflite::gpu::ObjectsAssignment<size_t> assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(usageRecords, MemoryStrategy::GREEDY_BEST, &assignment));
  auto objectsCount = assignment.object_sizes.size();
  std::vector<id<MTLBuffer>> sharedBuffers(objectsCount);
  const bool f32_storage = _precision == CalculationsPrecision::F32;
  size_t dataTypeSize = f32_storage ? sizeof(float) : sizeof(HalfBits);

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
      return absl::ResourceExhaustedError(error);
    }
#endif
#if defined(__MAC_10_12) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_12
    if ([_device currentAllocatedSize] + bufferSize > [_device recommendedMaxWorkingSetSize]) {
      std::string error("Out of memory in MTLBuffer allocation. Currently allocated: ");
      error += std::to_string([_device currentAllocatedSize]);
      return absl::ResourceExhaustedError(error);
    }
#endif

    sharedBuffers[i] = [_device newBufferWithLength:bufferSize
                                            options:MTLResourceStorageModeShared];
  }
  for (int i = 0; i < _computeTasks.size(); ++i) {
    auto& task = _computeTasks[i];
    if ([task hasInOutIds:preallocatedIds]) {
      _taskIdsWithInOutBuffers.push_back(i);
    }
    RETURN_IF_ERROR([task assignBuffers:&buffers
                              outputIds:_outputIds
                         usageRecordIds:usageRecordIds
                        sharedBufferIds:assignment.object_ids
                          sharedBuffers:sharedBuffers]);
  }
  return absl::OkStatus();
}

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
       inputOutputBuffers:
           (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers {
  for (auto& task_index : _taskIdsWithInOutBuffers) {
    auto& task = _computeTasks[task_index];
    [task updateBuffers:inputOutputBuffers];
  }
  for (int i = 0; i < _computeTasks.size(); ++i) {
    auto& task = _computeTasks[i];
    [task encodeWithEncoder:commandEncoder];
  }
}

- (void)encodeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
             inputOutputBuffers:
                 (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers {
  for (auto& task_index : _taskIdsWithInOutBuffers) {
    auto& task = _computeTasks[task_index];
    [task updateBuffers:inputOutputBuffers];
  }
  for (int i = 0; i < _computeTasks.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    auto& task = _computeTasks[i];
    [task encodeWithEncoder:encoder];
    [encoder endEncoding];
  }
}

- (void)encodeWithCommandQueue:(id<MTLCommandQueue>)commandQueue
            inputOutputBuffers:
                (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers
             flushPeriodically:(int)flushPeriod {
  for (auto& task_index : _taskIdsWithInOutBuffers) {
    auto& task = _computeTasks[task_index];
    [task updateBuffers:inputOutputBuffers];
  }
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  for (int i = 0; i < _computeTasks.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    auto& task = _computeTasks[i];
    [task encodeWithEncoder:encoder];
    [encoder endEncoding];
    if (i % flushPeriod == (flushPeriod - 1)) {
      [commandBuffer commit];
      commandBuffer = [commandQueue commandBuffer];
    }
  }
  [commandBuffer commit];
}

@end
