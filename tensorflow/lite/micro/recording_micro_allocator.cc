/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/recording_micro_allocator.h"

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/recording_simple_memory_allocator.h"

namespace tflite {

RecordingMicroAllocator::RecordingMicroAllocator(
    RecordingSimpleMemoryAllocator* recording_memory_allocator,
    ErrorReporter* error_reporter)
    : MicroAllocator(recording_memory_allocator, error_reporter),
      recording_memory_allocator_(recording_memory_allocator) {}

RecordingMicroAllocator* RecordingMicroAllocator::Create(
    uint8_t* tensor_arena, size_t arena_size, ErrorReporter* error_reporter) {
  TFLITE_DCHECK(error_reporter != nullptr);

  RecordingSimpleMemoryAllocator* simple_memory_allocator =
      RecordingSimpleMemoryAllocator::Create(error_reporter, tensor_arena,
                                             arena_size);
  TFLITE_DCHECK(simple_memory_allocator != nullptr);

  uint8_t* allocator_buffer = simple_memory_allocator->AllocateFromTail(
      sizeof(RecordingMicroAllocator), alignof(RecordingMicroAllocator));
  RecordingMicroAllocator* allocator = new (allocator_buffer)
      RecordingMicroAllocator(simple_memory_allocator, error_reporter);
  return allocator;
}

RecordedAllocation RecordingMicroAllocator::GetRecordedAllocation(
    RecordedAllocationType allocation_type) const {
  switch (allocation_type) {
    case RecordedAllocationType::kTfLiteEvalTensorData:
      return recorded_tflite_eval_tensor_data_;
    case RecordedAllocationType::kPersistentTfLiteTensorData:
      return recorded_persistent_tflite_tensor_data_;
    case RecordedAllocationType::kPersistentTfLiteTensorQuantizationData:
      return recorded_persistent_tflite_tensor_quantization_data_;
    case RecordedAllocationType::kPersistentBufferData:
      return recorded_persistent_buffer_data_;
    case RecordedAllocationType::kTfLiteTensorVariableBufferData:
      return recorded_tflite_tensor_variable_buffer_data_;
    case RecordedAllocationType::kNodeAndRegistrationArray:
      return recorded_node_and_registration_array_data_;
    case RecordedAllocationType::kOpData:
      return recorded_op_data_;
  }
  TF_LITE_REPORT_ERROR(error_reporter(), "Invalid allocation type supplied: %d",
                       allocation_type);
  return RecordedAllocation();
}

const RecordingSimpleMemoryAllocator*
RecordingMicroAllocator::GetSimpleMemoryAllocator() const {
  return recording_memory_allocator_;
}

void RecordingMicroAllocator::PrintAllocations() const {
  TF_LITE_REPORT_ERROR(
      error_reporter(),
      "[RecordingMicroAllocator] Arena allocation total %d bytes",
      recording_memory_allocator_->GetUsedBytes());
  TF_LITE_REPORT_ERROR(
      error_reporter(),
      "[RecordingMicroAllocator] Arena allocation head %d bytes",
      recording_memory_allocator_->GetHeadUsedBytes());
  TF_LITE_REPORT_ERROR(
      error_reporter(),
      "[RecordingMicroAllocator] Arena allocation tail %d bytes",
      recording_memory_allocator_->GetTailUsedBytes());
  PrintRecordedAllocation(RecordedAllocationType::kTfLiteEvalTensorData,
                          "TfLiteEvalTensor data", "allocations");
  PrintRecordedAllocation(RecordedAllocationType::kPersistentTfLiteTensorData,
                          "Persistent TfLiteTensor data", "tensors");
  PrintRecordedAllocation(
      RecordedAllocationType::kPersistentTfLiteTensorQuantizationData,
      "Persistent TfLiteTensor quantization data", "allocations");
  PrintRecordedAllocation(RecordedAllocationType::kPersistentBufferData,
                          "Persistent buffer data", "allocations");
  PrintRecordedAllocation(
      RecordedAllocationType::kTfLiteTensorVariableBufferData,
      "TfLiteTensor variable buffer data", "allocations");
  PrintRecordedAllocation(RecordedAllocationType::kNodeAndRegistrationArray,
                          "NodeAndRegistration struct",
                          "NodeAndRegistration structs");
  PrintRecordedAllocation(RecordedAllocationType::kOpData,
                          "Operator runtime data", "OpData structs");
}

void* RecordingMicroAllocator::AllocatePersistentBuffer(size_t bytes) {
  RecordedAllocation allocations = SnapshotAllocationUsage();
  void* buffer = MicroAllocator::AllocatePersistentBuffer(bytes);
  RecordAllocationUsage(allocations, recorded_persistent_buffer_data_);

  return buffer;
}

void RecordingMicroAllocator::PrintRecordedAllocation(
    RecordedAllocationType allocation_type, const char* allocation_name,
    const char* allocation_description) const {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  RecordedAllocation allocation = GetRecordedAllocation(allocation_type);
  if (allocation.used_bytes > 0 || allocation.requested_bytes > 0) {
    TF_LITE_REPORT_ERROR(
        error_reporter(),
        "[RecordingMicroAllocator] '%s' used %d bytes with alignment overhead "
        "(requested %d bytes for %d %s)",
        allocation_name, allocation.used_bytes, allocation.requested_bytes,
        allocation.count, allocation_description);
  }
#endif
}

TfLiteStatus RecordingMicroAllocator::AllocateNodeAndRegistrations(
    const Model* model, NodeAndRegistration** node_and_registrations) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status = MicroAllocator::AllocateNodeAndRegistrations(
      model, node_and_registrations);

  RecordAllocationUsage(allocations,
                        recorded_node_and_registration_array_data_);
  // The allocation count in SimpleMemoryAllocator will only be 1. To provide
  // better logging, decrement by 1 and add in the actual number of operators
  // used in the graph:
  // The allocation for this recording will always be 1. This is because the
  // parent class mallocs one large allocation for the number of nodes in the
  // graph (e.g. sizeof(NodeAndRegistration) * num_nodes).
  // To prevent extra overhead and potential for fragmentation, manually adjust
  // the accounting by decrementing by 1 and adding the actual number of nodes
  // used in the graph:
  recorded_node_and_registration_array_data_.count +=
      GetSubGraphFromModel(model)->operators()->size() - 1;
  return status;
}

TfLiteStatus
RecordingMicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
    const Model* model, const MicroOpResolver& op_resolver,
    NodeAndRegistration* node_and_registrations) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status =
      MicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
          model, op_resolver, node_and_registrations);

  RecordAllocationUsage(allocations, recorded_op_data_);
  return status;
}

TfLiteStatus RecordingMicroAllocator::AllocateTfLiteEvalTensors(
    const Model* model, TfLiteEvalTensor** eval_tensors) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status =
      MicroAllocator::AllocateTfLiteEvalTensors(model, eval_tensors);

  RecordAllocationUsage(allocations, recorded_tflite_eval_tensor_data_);
  // The allocation for this recording will always be 1. This is because the
  // parent class mallocs one large allocation for the number of tensors in the
  // graph (e.g. sizeof(TfLiteEvalTensor) * num_tensors).
  // To prevent extra overhead and potential for fragmentation, manually adjust
  // the accounting by decrementing by 1 and adding the actual number of tensors
  // used in the graph:
  recorded_tflite_eval_tensor_data_.count +=
      GetSubGraphFromModel(model)->tensors()->size() - 1;
  return status;
}

TfLiteStatus RecordingMicroAllocator::AllocateVariables(
    const SubGraph* subgraph, TfLiteEvalTensor* eval_tensors) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status =
      MicroAllocator::AllocateVariables(subgraph, eval_tensors);

  RecordAllocationUsage(allocations,
                        recorded_tflite_tensor_variable_buffer_data_);
  return status;
}

TfLiteTensor* RecordingMicroAllocator::AllocatePersistentTfLiteTensorInternal(
    const Model* model, TfLiteEvalTensor* eval_tensors, int tensor_index) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteTensor* result = MicroAllocator::AllocatePersistentTfLiteTensorInternal(
      model, eval_tensors, tensor_index);

  RecordAllocationUsage(allocations, recorded_persistent_tflite_tensor_data_);
  return result;
}

TfLiteStatus RecordingMicroAllocator::PopulateTfLiteTensorFromFlatbuffer(
    const Model* model, const SubGraph* subgraph, TfLiteTensor* tensor,
    int tensor_index, bool allocate_temp) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status = MicroAllocator::PopulateTfLiteTensorFromFlatbuffer(
      model, subgraph, tensor, tensor_index, allocate_temp);

  RecordAllocationUsage(allocations,
                        recorded_persistent_tflite_tensor_quantization_data_);
  return status;
}

RecordedAllocation RecordingMicroAllocator::SnapshotAllocationUsage() const {
  return {/*requested_bytes=*/recording_memory_allocator_->GetRequestedBytes(),
          /*used_bytes=*/recording_memory_allocator_->GetUsedBytes(),
          /*count=*/recording_memory_allocator_->GetAllocatedCount()};
}

void RecordingMicroAllocator::RecordAllocationUsage(
    const RecordedAllocation& snapshotted_allocation,
    RecordedAllocation& recorded_allocation) {
  recorded_allocation.requested_bytes +=
      recording_memory_allocator_->GetRequestedBytes() -
      snapshotted_allocation.requested_bytes;
  recorded_allocation.used_bytes +=
      recording_memory_allocator_->GetUsedBytes() -
      snapshotted_allocation.used_bytes;
  recorded_allocation.count +=
      recording_memory_allocator_->GetAllocatedCount() -
      snapshotted_allocation.count;
}

}  // namespace tflite
