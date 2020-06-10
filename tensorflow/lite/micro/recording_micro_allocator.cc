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
    TfLiteContext* context, const Model* model,
    RecordingSimpleMemoryAllocator* recording_memory_allocator,
    ErrorReporter* error_reporter)
    : MicroAllocator(context, model, recording_memory_allocator,
                     error_reporter),
      recording_memory_allocator_(recording_memory_allocator) {}

RecordingMicroAllocator* RecordingMicroAllocator::Create(
    TfLiteContext* context, const Model* model,
    RecordingSimpleMemoryAllocator* memory_allocator,
    ErrorReporter* error_reporter) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(model != nullptr);
  TFLITE_DCHECK(memory_allocator != nullptr);
  uint8_t* allocator_buffer = memory_allocator->AllocateFromTail(
      sizeof(RecordingMicroAllocator), alignof(RecordingMicroAllocator));
  RecordingMicroAllocator* allocator = new (allocator_buffer)
      RecordingMicroAllocator(context, model, memory_allocator, error_reporter);
  if (allocator->InitGraphAndContextTensorData() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "RecordingMicroAllocator: Failed to initialize model graph.");
    return nullptr;
  }
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

TfLiteStatus RecordingMicroAllocator::AllocateTfLiteTensorArray() {
  SnapshotAllocationUsage(recorded_tflite_tensor_array_data_);

  TfLiteStatus status = MicroAllocator::AllocateTfLiteTensorArray();

  RecordAllocationUsage(recorded_tflite_tensor_array_data_);
  recorded_tflite_tensor_array_data_.count = GetTensorsCount();
  return status;
}

TfLiteStatus
RecordingMicroAllocator::PopulateTfLiteTensorArrayFromFlatbuffer() {
  SnapshotAllocationUsage(recorded_tflite_tensor_array_quantization_data_);

  TfLiteStatus status =
      MicroAllocator::PopulateTfLiteTensorArrayFromFlatbuffer();

  RecordAllocationUsage(recorded_tflite_tensor_array_quantization_data_);
  return status;
}

TfLiteStatus RecordingMicroAllocator::AllocateNodeAndRegistrations(
    NodeAndRegistration** node_and_registrations) {
  SnapshotAllocationUsage(recorded_node_and_registration_array_data_);

  TfLiteStatus status =
      MicroAllocator::AllocateNodeAndRegistrations(node_and_registrations);

  RecordAllocationUsage(recorded_node_and_registration_array_data_);
  recorded_node_and_registration_array_data_.count = GetOperatorsCount();
  return status;
}

TfLiteStatus
RecordingMicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
    const MicroOpResolver& op_resolver,
    NodeAndRegistration* node_and_registrations) {
  SnapshotAllocationUsage(recorded_op_data_);

  TfLiteStatus status =
      MicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
          op_resolver, node_and_registrations);

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
