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

#ifndef TENSORFLOW_LITE_MICRO_RECORDING_MICRO_ALLOCATOR_H_
#define TENSORFLOW_LITE_MICRO_RECORDING_MICRO_ALLOCATOR_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/recording_simple_memory_allocator.h"

namespace tflite {

// List of buckets currently recorded by this class. Each type keeps a list of
// allocated information during model initialization.
enum class RecordedAllocationType {
  kTfLiteTensorArray,
  kTfLiteTensorArrayQuantizationData,
  kNodeAndRegistrationArray,
  kOpData
};

// Container for holding information about allocation recordings by a given
// type. Each recording contains the number of bytes requested, the actual bytes
// allocated (can defer from requested by alignment), and the number of items
// allocated.
typedef struct RecordedAllocation {
  RecordedAllocation() : requested_bytes(0), used_bytes(0), count(0) {}
  size_t requested_bytes;
  size_t used_bytes;
  size_t count;
} RecordedAllocation;

// Utility subclass of MicroAllocator that records all allocations
// inside the arena. A summary of allocations can be logged through the
// ErrorReporter by invoking LogAllocations(). Individual allocation recordings
// can be retrieved by type through the GetRecordedAllocation() function. This
// class should only be used for auditing memory usage or integration testing.
class RecordingMicroAllocator : public MicroAllocator {
 public:
  RecordingMicroAllocator(TfLiteContext* context, const Model* model,
                          RecordingSimpleMemoryAllocator* memory_allocator,
                          ErrorReporter* error_reporter);

  // Returns the recorded allocations information for a given allocation type.
  RecordedAllocation GetRecordedAllocation(
      RecordedAllocationType allocation_type);

  // Logs out through the ErrorReporter all allocation recordings by type
  // defined in RecordedAllocationType.
  void PrintAllocations();

 protected:
  TfLiteStatus AllocateTfLiteTensorArray() override;
  TfLiteStatus PopulateTfLiteTensorArrayFromFlatbuffer() override;
  TfLiteStatus AllocateNodeAndRegistrations(
      NodeAndRegistration** node_and_registrations) override;
  TfLiteStatus PrepareNodeAndRegistrationDataFromFlatbuffer(
      const MicroOpResolver& op_resolver,
      NodeAndRegistration* node_and_registrations) override;

  void SnapshotAllocationUsage(RecordedAllocation& recorded_allocation);
  void RecordAllocationUsage(RecordedAllocation& recorded_allocation);

 private:
  void PrintRecordedAllocation(RecordedAllocationType allocation_type,
                               const char* allocation_name);

  RecordingSimpleMemoryAllocator* recording_memory_allocator_;

  RecordedAllocation recorded_tflite_tensor_array_data_;
  RecordedAllocation recorded_tflite_tensor_array_quantization_data_;
  RecordedAllocation recorded_node_and_registration_array_data_;
  RecordedAllocation recorded_op_data_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_RECORDING_MICRO_ALLOCATOR_H_
