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

namespace tflite {
namespace gpu {
namespace metal {
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

absl::Status InferenceContext::CompileModelWithDevice(
      id<MTLDevice> device, const CompiledModel& compiled_model,
      const std::vector<ValueId>& input_ids,
      const std::vector<ValueId>& output_ids,
      CalculationsPrecision precision) {
  input_ids_ = input_ids;
  output_ids_ = output_ids;
  precision_ = precision;
  // Metal resources are created here.
  for (const auto& node : compiled_model.nodes) {
    ComputeTask task;
    RETURN_IF_ERROR(task.CompileWithDevice(device, node, precision_));
    task.SetDescription(node.description);
    compute_tasks_.emplace_back(std::move(task));
  }
  tensor_shapes_ = compiled_model.tensor_shapes;
  for (auto& task : compute_tasks_) {
    // The same device must be used here as well as on shader compilation stage.
    RETURN_IF_ERROR(task.UpdateParamsWithDevice(device, tensor_shapes_));
  }
  RETURN_IF_ERROR(AllocateTensors(device));
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateTensors(id<MTLDevice> device) {
  std::set<ValueId> preallocated_ids;
  for (auto tensor_id : input_ids_) {
    preallocated_ids.insert(tensor_id);
  }
  for (const auto& outputId : output_ids_) {
    preallocated_ids.insert(outputId);
  }
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    auto& task = compute_tasks_[i];
    if (task.HasInOutIds(preallocated_ids)) {
      task_ids_with_preallocated_tensors_.push_back(i);
    }
  }

  const bool f32_storage = precision_ == CalculationsPrecision::F32;
  for (auto& tensor_id : preallocated_ids) {
    BHWC shape = tensor_shapes_[tensor_id];
    TensorDescriptor descriptor;
    descriptor.storage_type = TensorStorageType::BUFFER;
    descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
    descriptor.layout = Layout::HWC;
    preallocated_tensors_[tensor_id] =
        CreateSharedBufferTensor(nil, shape, descriptor);
  }

  RETURN_IF_ERROR(AllocateMemoryForBuffers(device));
  BindTensorsToOperations();
  return absl::OkStatus();
}

MetalSpatialTensor* InferenceContext::GetTensor(ValueId tensor_id) {
  if (preallocated_tensors_.find(tensor_id) != preallocated_tensors_.end()) {
    return &preallocated_tensors_[tensor_id];
  } else if (graph_ids_to_shared_buffer_tensors_.find(tensor_id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_[graph_ids_to_shared_buffer_tensors_[tensor_id]];
  }
  return nullptr;
}

void InferenceContext::BindTensorsToOperations() {
  for (auto& task : compute_tasks_) {
    const auto& src_ids = task.GetInputIds();
    for (int i = 0; i < src_ids.size(); ++i) {
      MetalSpatialTensor* tensor = GetTensor(src_ids[i]);
      task.SetSrcTensor(*tensor, i);
    }
    const auto& dst_ids = task.GetOutputIds();
    for (int i = 0; i < dst_ids.size(); ++i) {
      MetalSpatialTensor* tensor = GetTensor(dst_ids[i]);
      task.SetDstTensor(*tensor, i);
    }
  }
}

void InferenceContext::GetUsages(std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (preallocated_tensors_.find(in_id) == preallocated_tensors_.end()) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < compute_tasks_.size(); ++op_index) {
    for (auto& tensor_id : compute_tasks_[op_index].GetInputIds()) {
      if (preallocated_tensors_.find(tensor_id) == preallocated_tensors_.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
    for (auto& tensor_id : compute_tasks_[op_index].GetOutputIds()) {
      if (preallocated_tensors_.find(tensor_id) == preallocated_tensors_.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (preallocated_tensors_.find(out_id) == preallocated_tensors_.end()) {
      AddUsage(out_id, compute_tasks_.size(), usages);
    }
  }
}

absl::Status InferenceContext::AllocateMemoryForBuffers(id<MTLDevice> device) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(&buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& shape = tensor_shapes_[usage.first];
    const size_t buffer_size =
        shape.b * shape.w * shape.h * AlignByN(shape.c, 4);
    graph_ids_to_shared_buffer_tensors_[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  const bool f32_storage = precision_ == CalculationsPrecision::F32;
  size_t dataTypeSize = f32_storage ? sizeof(float) : sizeof(HalfBits);
  shared_buffers_.resize(buffer_assignment.object_sizes.size());
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

    shared_buffers_[i] = [device newBufferWithLength:bufferSize
                                            options:MTLResourceStorageModeShared];
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  TensorDescriptor descriptor;
  descriptor.storage_type = TensorStorageType::BUFFER;
  descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
  descriptor.layout = Layout::HWC;
  for (auto& task : compute_tasks_) {
    const std::vector<ValueId> input_ids = task.GetInputIds();
    const std::vector<ValueId> output_ids = task.GetOutputIds();
    std::vector<ValueId> all_ids = input_ids;
    all_ids.insert(all_ids.end(), output_ids.begin(), output_ids.end());
    for (auto& tensor_id : all_ids) {
      if (preallocated_tensors_.find(tensor_id) != preallocated_tensors_.end()) continue;
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const auto& shape = tensor_shapes_[tensor_id];
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      shared_buffer_tensors_[tensor_index] = CreateSharedBufferTensor(
                                           shared_buffers_[buffer_index], shape, descriptor);
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

void InferenceContext::EncodeWithEncoder(
      id<MTLComputeCommandEncoder> command_encoder,
      const std::map<ValueId, id<MTLBuffer>>& in_out_buffers) {
  UpdatePreallocatedTensors(in_out_buffers);
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    auto& task = compute_tasks_[i];
    task.EncodeWithEncoder(command_encoder);
  }
}

void InferenceContext::EncodeWithCommandBuffer(
      id<MTLCommandBuffer> command_buffer,
      const std::map<ValueId, id<MTLBuffer>>& in_out_buffers) {
  UpdatePreallocatedTensors(in_out_buffers);
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    auto& task = compute_tasks_[i];
    task.EncodeWithEncoder(encoder);
    [encoder endEncoding];
  }
}

void InferenceContext::EncodeWithCommandQueue(
      id<MTLCommandQueue> command_queue,
      const std::map<ValueId, id<MTLBuffer>>& in_out_buffers,
      int flush_period) {
  UpdatePreallocatedTensors(in_out_buffers);
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    auto& task = compute_tasks_[i];
    task.EncodeWithEncoder(encoder);
    [encoder endEncoding];
    if (i % flush_period == (flush_period - 1)) {
      [command_buffer commit];
      command_buffer = [command_queue commandBuffer];
    }
  }
  [command_buffer commit];
}

void InferenceContext::UpdatePreallocatedTensors(
    const std::map<ValueId, id<MTLBuffer>>& preallocated) {
  for (const auto& it : preallocated) {
    preallocated_tensors_[it.first].SetBufferHandle(it.second);
  }
  for (auto& task_index : task_ids_with_preallocated_tensors_) {
    auto& task = compute_tasks_[task_index];
    const auto& src_ids = task.GetInputIds();
    for (int i = 0; i < src_ids.size(); ++i) {
      const auto& it = preallocated_tensors_.find(src_ids[i]);
      if (it != preallocated_tensors_.end()) {
        task.SetSrcTensor(it->second, i);
      }
    }
    const auto& dst_ids = task.GetOutputIds();
    for (int i = 0; i < dst_ids.size(); ++i) {
      const auto& it = preallocated_tensors_.find(dst_ids[i]);
      if (it != preallocated_tensors_.end()) {
        task.SetDstTensor(it->second, i);
      }
    }
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
