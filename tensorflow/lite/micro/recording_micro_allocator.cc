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
#include "tensorflow/lite/micro/recording_simple_memory_allocator.h"

namespace tflite {

RecordingMicroAllocator::RecordingMicroAllocator(
    RecordingSimpleMemoryAllocator* recording_memory_allocator,
    ErrorReporter* error_reporter)
    : MicroAllocator(recording_memory_allocator, error_reporter),
      recording_memory_allocator_(recording_memory_allocator) {}

RecordingMicroAllocator* RecordingMicroAllocator::Create(
    RecordingSimpleMemoryAllocator* memory_allocator,
    ErrorReporter* error_reporter) {
  TFLITE_DCHECK(memory_allocator != nullptr);
  uint8_t* allocator_buffer = memory_allocator->AllocateFromTail(
      sizeof(RecordingMicroAllocator), alignof(RecordingMicroAllocator));
  RecordingMicroAllocator* allocator = new (allocator_buffer)
      RecordingMicroAllocator(memory_allocator, error_reporter);
  return allocator;
}

RecordedAllocation RecordingMicroAllocator::GetRecordedAllocation(
    RecordedAllocationType allocation_type) {
  switch (allocation_type) {
    case RecordedAllocationType::kTfLiteTensorArray:
      return recorded_tflite_tensor_array_data_;
    case RecordedAllocationType::kTfLiteTensorArrayQuantizationData:
      return recorded_tflite_tensor_array_quantization_data_;
    case RecordedAllocationType::kNodeAndRegistrationArray:
      return recorded_node_and_registration_array_data_;
    case RecordedAllocationType::kOpData:
      return recorded_op_data_;
  }
  TF_LITE_REPORT_ERROR(error_reporter(), "Invalid allocation type supplied: %d",
                       allocation_type);
  return RecordedAllocation();
}

void RecordingMicroAllocator::PrintAllocations() {
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
  PrintRecordedAllocation(RecordedAllocationType::kTfLiteTensorArray,
                          "TfLiteTensor struct allocation");
  PrintRecordedAllocation(
      RecordedAllocationType::kTfLiteTensorArrayQuantizationData,
      "TfLiteTensor quantization data allocations");
  PrintRecordedAllocation(RecordedAllocationType::kNodeAndRegistrationArray,
                          "NodeAndRegistration struct allocation");
  PrintRecordedAllocation(RecordedAllocationType::kOpData,
                          "Operator runtime data allocation");
}

void RecordingMicroAllocator::PrintRecordedAllocation(
    RecordedAllocationType allocation_type, const char* allocation_name) {
  RecordedAllocation allocation = GetRecordedAllocation(allocation_type);
  TF_LITE_REPORT_ERROR(error_reporter(),
                       "[RecordingMicroAllocator] '%s' used %d bytes "
                       "(requested %d bytes %d times)",
                       allocation_name, allocation.used_bytes,
                       allocation.requested_bytes, allocation.count);
}

TfLiteStatus RecordingMicroAllocator::AllocateTfLiteTensorArray(
    TfLiteContext* context, const SubGraph* subgraph) {
  SnapshotAllocationUsage(recorded_tflite_tensor_array_data_);

  TfLiteStatus status =
      MicroAllocator::AllocateTfLiteTensorArray(context, subgraph);

  RecordAllocationUsage(recorded_tflite_tensor_array_data_);
  recorded_tflite_tensor_array_data_.count = context->tensors_size;
  return status;
}

TfLiteStatus RecordingMicroAllocator::PopulateTfLiteTensorArrayFromFlatbuffer(
    const Model* model, TfLiteContext* context, const SubGraph* subgraph) {
  SnapshotAllocationUsage(recorded_tflite_tensor_array_quantization_data_);

  TfLiteStatus status = MicroAllocator::PopulateTfLiteTensorArrayFromFlatbuffer(
      model, context, subgraph);

  RecordAllocationUsage(recorded_tflite_tensor_array_quantization_data_);
  return status;
}

TfLiteStatus RecordingMicroAllocator::AllocateNodeAndRegistrations(
    const SubGraph* subgraph, NodeAndRegistration** node_and_registrations) {
  SnapshotAllocationUsage(recorded_node_and_registration_array_data_);

  TfLiteStatus status = MicroAllocator::AllocateNodeAndRegistrations(
      subgraph, node_and_registrations);

  RecordAllocationUsage(recorded_node_and_registration_array_data_);
  recorded_node_and_registration_array_data_.count =
      subgraph->operators()->size();
  return status;
}

TfLiteStatus
RecordingMicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
    const Model* model, const SubGraph* subgraph,
    const MicroOpResolver& op_resolver,
    NodeAndRegistration* node_and_registrations) {
  SnapshotAllocationUsage(recorded_op_data_);

  TfLiteStatus status =
      MicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
          model, subgraph, op_resolver, node_and_registrations);

  RecordAllocationUsage(recorded_op_data_);
  return status;
}

void RecordingMicroAllocator::SnapshotAllocationUsage(
    RecordedAllocation& recorded_allocation) {
  recorded_allocation.requested_bytes =
      recording_memory_allocator_->GetRequestedBytes();
  recorded_allocation.used_bytes = recording_memory_allocator_->GetUsedBytes();
  recorded_allocation.count = recording_memory_allocator_->GetAllocatedCount();
}

void RecordingMicroAllocator::RecordAllocationUsage(
    RecordedAllocation& recorded_allocation) {
  recorded_allocation.requested_bytes =
      recording_memory_allocator_->GetRequestedBytes() -
      recorded_allocation.requested_bytes;
  recorded_allocation.used_bytes = recording_memory_allocator_->GetUsedBytes() -
                                   recorded_allocation.used_bytes;
  recorded_allocation.count = recording_memory_allocator_->GetAllocatedCount() -
                              recorded_allocation.count;
}

}  // namespace tflite
