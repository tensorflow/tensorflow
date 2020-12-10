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
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

using ::tflite::gpu::AlignByN;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::CalculationsPrecision;
using ::tflite::gpu::DataType;
using ::tflite::gpu::HalfBits;
using ::tflite::gpu::int2;
using ::tflite::gpu::MemoryStrategy;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::metal::MetalSpatialTensor;
using ::tflite::gpu::TensorDescriptor;
using ::tflite::gpu::TensorStorageType;
using ::tflite::gpu::TensorUsageRecord;
using ::tflite::gpu::ValueId;

namespace {
void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    // initializing start index(.x) and end index(.y)
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    // updating end index(.y)
    (*usage_records)[id].y = task_index;
  }
}
}  // namespace

@implementation TFLInferenceContext {
  std::vector<TFLComputeTask*> _computeTasks;
  // contains indexes of _computeTasks
  std::vector<int> _taskIdsWithPreallocatedTensors;
  std::vector<ValueId> _inputIds;
  std::vector<ValueId> _outputIds;
  CalculationsPrecision _precision;
  std::map<ValueId, BHWC> _tensorShapes;
  std::map<ValueId, MetalSpatialTensor> _preallocatedTensors;

  std::map<ValueId, int> _graphIdsToSharedBufferTensors;
  std::vector<id<MTLBuffer>> _sharedBuffers;
  std::vector<MetalSpatialTensor> _sharedBufferTensors;  // use references to memory
                                                         // from _sharedBuffers
}

- (absl::Status)compileModelWithDevice:(id<MTLDevice>)device
                                 model:(const tflite::gpu::metal::CompiledModel&) compiledModel
                        inputBufferIDs:(const std::vector<tflite::gpu::ValueId>&)inputBufferIDs
                       outputBufferIDs:(const std::vector<tflite::gpu::ValueId>&)outputBufferIDs
                             precision:(tflite::gpu::CalculationsPrecision)precision {
  _inputIds = inputBufferIDs;
  _outputIds = outputBufferIDs;
  _precision = precision;
  // Metal resources are created here.
  for (const auto& node : compiledModel.nodes) {
    TFLComputeTask* task = [[TFLComputeTask alloc] init];
    RETURN_IF_ERROR([task compileWithDevice:device
                             taskDescriptor:node
                                  precision:_precision]);
    [task setDescription:node.description];
    _computeTasks.emplace_back(task);
  }
  _tensorShapes = compiledModel.tensor_shapes;
  for (auto& task : _computeTasks) {
    // The same device must be used here as well as on shader compilation stage.
    RETURN_IF_ERROR([task updateParamsWithDevice:device tensorShapes:_tensorShapes]);
  }
  RETURN_IF_ERROR([self allocateTensors:device]);
  return absl::OkStatus();
}

- (absl::Status)allocateTensors:(id<MTLDevice>)device {
  std::set<ValueId> preallocatedIds;
  for (auto tensor_id : _inputIds) {
    preallocatedIds.insert(tensor_id);
  }
  for (const auto& outputId : _outputIds) {
    preallocatedIds.insert(outputId);
  }
  for (int i = 0; i < _computeTasks.size(); ++i) {
    auto& task = _computeTasks[i];
    if ([task hasInOutIds:preallocatedIds]) {
      _taskIdsWithPreallocatedTensors.push_back(i);
    }
  }

  const bool f32_storage = _precision == CalculationsPrecision::F32;
  for (auto& tensor_id : preallocatedIds) {
    BHWC shape = _tensorShapes[tensor_id];
    TensorDescriptor descriptor;
    descriptor.storage_type = TensorStorageType::BUFFER;
    descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
    descriptor.layout = tflite::gpu::Layout::HWC;
    _preallocatedTensors[tensor_id] =
        tflite::gpu::metal::CreateSharedBufferTensor(nil, shape, descriptor);
  }

  RETURN_IF_ERROR([self allocateMemoryForBuffers:device]);
  [self bindTensorsToOperations];
  return absl::OkStatus();
}

- (MetalSpatialTensor*)getTensor:(ValueId)tensorId {
  if (_preallocatedTensors.find(tensorId) != _preallocatedTensors.end()) {
    return &_preallocatedTensors[tensorId];
  } else if (_graphIdsToSharedBufferTensors.find(tensorId) !=
             _graphIdsToSharedBufferTensors.end()) {
    return &_sharedBufferTensors[_graphIdsToSharedBufferTensors[tensorId]];
  }
  return nullptr;
}

- (void)bindTensorsToOperations {
  for (auto& task : _computeTasks) {
    const auto& src_ids = [task getInputIds];
    for (int i = 0; i < src_ids.size(); ++i) {
      MetalSpatialTensor* tensor = [self getTensor:src_ids[i]];
      [task setSrcTensor:*tensor withIndex:i];
    }
    const auto& dst_ids = [task getOutputIds];
    for (int i = 0; i < dst_ids.size(); ++i) {
      MetalSpatialTensor* tensor = [self getTensor:dst_ids[i]];
      [task setDstTensor:*tensor withIndex:i];
    }
  }
}

- (void)getUsages:(std::map<ValueId, int2>*) usages {
  for (ValueId in_id : _inputIds) {
    if (_preallocatedTensors.find(in_id) == _preallocatedTensors.end()) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < _computeTasks.size(); ++op_index) {
    for (auto& tensor_id : [_computeTasks[op_index] getInputIds]) {
      if (_preallocatedTensors.find(tensor_id) == _preallocatedTensors.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
    for (auto& tensor_id : [_computeTasks[op_index] getOutputIds]) {
      if (_preallocatedTensors.find(tensor_id) == _preallocatedTensors.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
  }
  for (ValueId out_id : _outputIds) {
    if (_preallocatedTensors.find(out_id) == _preallocatedTensors.end()) {
      AddUsage(out_id, _computeTasks.size(), usages);
    }
  }
}

- (absl::Status)allocateMemoryForBuffers:(id<MTLDevice>)device {
  std::map<ValueId, int2> buffer_usages;
  [self getUsages:&buffer_usages];

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& shape = _tensorShapes[usage.first];
    const size_t buffer_size =
        shape.b * shape.w * shape.h * AlignByN(shape.c, 4);
    _graphIdsToSharedBufferTensors[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<tflite::gpu::TaskId>(usage.second.x),
                                    static_cast<tflite::gpu::TaskId>(usage.second.y)});
  }

  tflite::gpu::ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  const bool f32_storage = _precision == CalculationsPrecision::F32;
  size_t dataTypeSize = f32_storage ? sizeof(float) : sizeof(HalfBits);
  _sharedBuffers.resize(buffer_assignment.object_sizes.size());
  for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
    // Initialize metal buffer
    NSUInteger bufferSize = dataTypeSize * buffer_assignment.object_sizes[i];

#if (defined(__MAC_10_14) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_14) ||      \
    (defined(__IPHONE_12_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_12_0) || \
    (defined(__TVOS_12_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_12_0)
    if (bufferSize > [device maxBufferLength]) {
      std::string error("Tensor id: ");
      error += std::to_string(buffer_assignment.object_ids[i]) +
               " with size: " + std::to_string(bufferSize) +
               " exceeds MTLDevice maxBufferLength: " + std::to_string([device maxBufferLength]);
      return absl::ResourceExhaustedError(error);
    }
#endif
#if defined(__MAC_10_12) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_12
    if ([device currentAllocatedSize] + bufferSize > [device recommendedMaxWorkingSetSize]) {
      std::string error("Out of memory in MTLBuffer allocation. Currently allocated: ");
      error += std::to_string([device currentAllocatedSize]);
      return absl::ResourceExhaustedError(error);
    }
#endif

    _sharedBuffers[i] = [device newBufferWithLength:bufferSize
                                            options:MTLResourceStorageModeShared];
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  _sharedBufferTensors.resize(buffer_usage_records.size());
  TensorDescriptor descriptor;
  descriptor.storage_type = TensorStorageType::BUFFER;
  descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
  descriptor.layout = tflite::gpu::Layout::HWC;
  for (auto& task : _computeTasks) {
    const std::vector<ValueId> input_ids = [task getInputIds];
    const std::vector<ValueId> output_ids = [task getOutputIds];
    std::vector<ValueId> all_ids = input_ids;
    all_ids.insert(all_ids.end(), output_ids.begin(), output_ids.end());
    for (auto& tensor_id : all_ids) {
      if (_preallocatedTensors.find(tensor_id) != _preallocatedTensors.end()) continue;
      const int tensor_index = _graphIdsToSharedBufferTensors[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const auto& shape = _tensorShapes[tensor_id];
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      _sharedBufferTensors[tensor_index] = tflite::gpu::metal::CreateSharedBufferTensor(
                                           _sharedBuffers[buffer_index], shape, descriptor);
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
       inputOutputBuffers:
           (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers {
  [self updatePreallocatedTensors:inputOutputBuffers];
  for (int i = 0; i < _computeTasks.size(); ++i) {
    auto& task = _computeTasks[i];
    [task encodeWithEncoder:commandEncoder];
  }
}

- (void)encodeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
             inputOutputBuffers:
                 (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers {
  [self updatePreallocatedTensors:inputOutputBuffers];
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
  [self updatePreallocatedTensors:inputOutputBuffers];
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

- (absl::Status)updatePreallocatedTensors:(const std::map<ValueId, id<MTLBuffer>>&)preallocated {
  for (const auto& it : preallocated) {
    _preallocatedTensors[it.first].SetBufferHandle(it.second);
  }
  for (auto& task_index : _taskIdsWithPreallocatedTensors) {
    auto& task = _computeTasks[task_index];
    const auto& src_ids = [task getInputIds];
    for (int i = 0; i < src_ids.size(); ++i) {
      const auto& it = _preallocatedTensors.find(src_ids[i]);
      if (it != _preallocatedTensors.end()) {
        [task setSrcTensor:it->second withIndex:i];
      }
    }
    const auto& dst_ids = [task getOutputIds];
    for (int i = 0; i < dst_ids.size(); ++i) {
      const auto& it = _preallocatedTensors.find(dst_ids[i]);
      if (it != _preallocatedTensors.end()) {
        [task setDstTensor:it->second withIndex:i];
      }
    }
  }
  return absl::OkStatus();
}

@end
