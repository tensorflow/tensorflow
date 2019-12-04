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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ALLOCATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ALLOCATOR_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/experimental/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

typedef struct {
  TfLiteNode node;
  const TfLiteRegistration* registration;
} NodeAndRegistration;

// Allocator responsible for allocating memory for all intermediate tensors
// necessary to invoke a model.
class MicroAllocator {
 public:
  // The lifetime of the model, tensor allocator and error reporter must be at
  // least as long as that of the allocator object, since the allocator needs
  // them to be accessible during its entire lifetime.
  MicroAllocator(TfLiteContext* context, const Model* model,
                 uint8_t* tensor_arena, size_t arena_size,
                 ErrorReporter* error_reporter);

  // Specify a particular tensor as pre-allocated.  This means that this tensor
  // will internally point to the supplied buffer, and no new memory will be
  // provided.  The buffer must live at least as long as the allocator, since
  // the buffer will be used every time an op is invoked which uses the
  // specified tensor.  Most commonly this is useful when a platform-provided
  // DMA buffer is used as an input, and it is desirable to avoid unnecessarily
  // allocating a new buffer and copying from the DMA buffer. The user must
  // ensure the buffer is valid throughout each interpreter run, and is not
  // prematurely overwritten.
  TfLiteStatus RegisterPreallocatedInput(uint8_t* buffer, size_t input_index);

  // Sets up all of the data structure members for a runtime tensor based on the
  // contents of a serialized tensor. This method doesn't allocate any memory,
  // all allocations happen subsequently in AllocateTensors.
  TfLiteStatus InitializeRuntimeTensor(
      const tflite::Tensor& flatbuffer_tensor,
      const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
      ErrorReporter* error_reporter, TfLiteTensor* result,
      uint8_t* preallocated_buffer = nullptr);

  // Run through the model and allocate all necessary input, output and
  // intermediate tensors except for those already provided via calls to
  // registerPreallocatedInput.
  // WARNING: doing any allocation after calling is method has the risk of
  // corruption tensor data so this method is the last method to be called in
  // this class.
  TfLiteStatus FinishTensorAllocation();

  // Run through the model to allocate nodes and registrations. We need to keep
  // them for the entire life time of the model to allow persistent tensors.
  // This method needs to be called before FinishTensorAllocation method.
  TfLiteStatus AllocateNodeAndRegistrations(
      const OpResolver& op_resolver,
      NodeAndRegistration** node_and_registrations);

 private:
  template <class T>
  TfLiteStatus FlatBufferIntArrayToTfLiteIntArray(
      const flatbuffers::Vector<T>* flat_array, TfLiteIntArray** result);

 private:
  const Model* model_;
  SimpleMemoryAllocator memory_allocator_;
  ErrorReporter* error_reporter_;
  TfLiteContext* context_;
  uint8_t* arena_;
  size_t arena_size_;
  // Indicating if the allocator is ready for allocation.
  bool active_ = false;

  const SubGraph* subgraph_;
  const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators_;
  const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ALLOCATOR_H_
