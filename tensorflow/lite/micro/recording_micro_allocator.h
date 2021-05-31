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
// TODO(b/169834511): Add tracking for scratch buffer allocations.
enum class RecordedAllocationType {
  kTfLiteEvalTensorData,
  kPersistentTfLiteTensorData,
  kPersistentTfLiteTensorQuantizationData,
  kPersistentBufferData,
  kTfLiteTensorVariableBufferData,
  kNodeAndRegistrationArray,
  kOpData,
};

// Container for holding information about allocation recordings by a given
// type. Each recording contains the number of bytes requested, the actual bytes
// allocated (can defer from requested by alignment), and the number of items
// allocated.
struct RecordedAllocation {
  size_t requested_bytes;
  size_t used_bytes;
  size_t count;
};

// Utility subclass of MicroAllocator that records all allocations
// inside the arena. A summary of allocations can be logged through the
// ErrorReporter by invoking LogAllocations(). This special allocator requires
// an instance of RecordingSimpleMemoryAllocator to capture allocations in the
// head and tail. Arena allocation recording can be retrieved by type through
// the GetRecordedAllocation() function. This class should only be used for
// auditing memory usage or integration testing.
class RecordingMicroAllocator : public MicroAllocator {
 public:
  static RecordingMicroAllocator* Create(uint8_t* tensor_arena,
                                         size_t arena_size,
                                         ErrorReporter* error_reporter);

  // Returns the recorded allocations information for a given allocation type.
  RecordedAllocation GetRecordedAllocation(
      RecordedAllocationType allocation_type) const;

  const RecordingSimpleMemoryAllocator* GetSimpleMemoryAllocator() const;

  // Logs out through the ErrorReporter all allocation recordings by type
  // defined in RecordedAllocationType.
  void PrintAllocations() const;

  void* AllocatePersistentBuffer(size_t bytes) override;

 protected:
  TfLiteStatus AllocateNodeAndRegistrations(
      const Model* model, SubgraphAllocations* subgraph_allocations) override;
  TfLiteStatus AllocateTfLiteEvalTensors(
      const Model* model, SubgraphAllocations* subgraph_allocations) override;
  TfLiteStatus AllocateVariables(const SubGraph* subgraph,
                                 TfLiteEvalTensor* eval_tensors) override;
  // TODO(b/162311891): Once all kernels have been updated to the new API drop
  // this method. It is only used to record TfLiteTensor persistent allocations.
  TfLiteTensor* AllocatePersistentTfLiteTensorInternal() override;

  // TODO(b/162311891): Once all kernels have been updated to the new API drop
  // this function since all allocations for quantized data will take place in
  // the temp section.
  TfLiteStatus PopulateTfLiteTensorFromFlatbuffer(const Model* model,
                                                  TfLiteTensor* tensor,
                                                  int tensor_index,
                                                  int subgraph_index,
                                                  bool allocate_temp) override;

 private:
  RecordingMicroAllocator(RecordingSimpleMemoryAllocator* memory_allocator,
                          ErrorReporter* error_reporter);

  void PrintRecordedAllocation(RecordedAllocationType allocation_type,
                               const char* allocation_name,
                               const char* allocation_description) const;

  RecordedAllocation SnapshotAllocationUsage() const;
  void RecordAllocationUsage(const RecordedAllocation& snapshotted_allocation,
                             RecordedAllocation& recorded_allocation);

  const RecordingSimpleMemoryAllocator* recording_memory_allocator_;

  RecordedAllocation recorded_tflite_eval_tensor_data_ = {};
  RecordedAllocation recorded_persistent_tflite_tensor_data_ = {};
  RecordedAllocation recorded_persistent_tflite_tensor_quantization_data_ = {};
  RecordedAllocation recorded_persistent_buffer_data_ = {};
  RecordedAllocation recorded_tflite_tensor_variable_buffer_data_ = {};
  RecordedAllocation recorded_node_and_registration_array_data_ = {};

  // TODO(b/187993291): Re-enable OpData allocating tracking.
  RecordedAllocation recorded_op_data_ = {};

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_RECORDING_MICRO_ALLOCATOR_H_
