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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_ALLOCATOR_H_
#define TENSORFLOW_LITE_MICRO_MICRO_ALLOCATOR_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Namespace used for unittests.
namespace internal {

// Sets up all of the data structure members for a runtime tensor
// based on the contents of a serialized tensor.
TfLiteStatus InitializeRuntimeTensor(
    SimpleMemoryAllocator* allocator, const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result);

// A handle tracking scratch buffer allocation. This handle is created by
// `RequestScratchBufferInArena`. `data` field is populated in
// `FinishTensorAllocation` after static memory planning.
// TODO(b/150257460) As a future optimization, this struct could be replaced by
// a union, since once `data` is populated, `bytes` and `node_idx` is not
// needed.
typedef struct {
  // Pointer to the scratch buffer.
  uint8_t* data;
  // Number of bytes required by the buffer. The actual allocated size might be
  // greater than `bytes` due to buffer alignment.
  size_t bytes;
  // Node where the buffer is allocated for. This provides useful information to
  // determine the lifetime of the buffer. In AllocationInfo, this buffer will
  // have `before` = node_idx and `after` = node_idx.
  int node_idx;
} ScratchBufferHandle;
}  // namespace internal

typedef struct {
  TfLiteNode node;
  const TfLiteRegistration* registration;
} NodeAndRegistration;

// Allocator responsible for allocating memory for all intermediate tensors
// necessary to invoke a model.

// Memory layout to help understand how it works
// This information could change in the future version.
// ************** .memory_allocator->GetBuffer()
// Tensors/Scratch buffers (head)
// ************** .head_watermark
// unused memory
// ************** .memory_allocator->GetBuffer() + ->GetMaxBufferSize()
//                                               - ->GetDataSize()
// persistent area (tail)
// ************** .memory_allocator->GetBuffer() + ->GetMaxBufferSize()
class MicroAllocator {
 public:
  // The lifetime of the model, tensor allocator and error reporter must be at
  // least as long as that of the allocator object, since the allocator needs
  // them to be accessible during its entire lifetime.

  // Note: Please use __declspec(align(16)) to make sure tensor_arena is 16
  // bytes aligned, otherwise some head room will be wasted.
  MicroAllocator(TfLiteContext* context, const Model* model,
                 uint8_t* tensor_arena, size_t arena_size,
                 ErrorReporter* error_reporter);

  // Runs through the model and allocates all necessary input, output and
  // intermediate tensors.
  // WARNING: doing any allocation after calling this method has the risk of
  // corrupting tensor data so this method should be the last non-const method
  // called in this class.
  TfLiteStatus FinishTensorAllocation();

  // Returns the arena usage in bytes, only available after
  // `FinishTensorAllocation`. Otherwise, it will return 0.
  size_t used_bytes() const {
    if (active_) {
      return 0;
    }
    return memory_allocator_->GetUsedBytes();
  }

  // Run through the model to allocate nodes and registrations. We need to keep
  // them for the entire life time of the model to allow persistent tensors.
  // This method needs to be called before FinishTensorAllocation method.
  TfLiteStatus AllocateNodeAndRegistrations(
      const OpResolver& op_resolver,
      NodeAndRegistration** node_and_registrations);

  // Allocates persistent buffer which has the same life time as the allocator.
  // The memory is immediately available and is allocated from the tail of the
  // arena.
  TfLiteStatus AllocatePersistentBuffer(size_t bytes, void** ptr);

  // Register a scratch buffer of size `bytes` for Node with `node_id`.
  // This method only allocates a BufferHandle holding information for memory
  // planning. The buffer ptr is ready after `FinishTensorAllocation` and can
  // be retrieved by `GetScratchBuffer` method using the returned buffer_idx.
  // Note that there should be no tail allocation between two consecutive
  // `RequestScratchBufferInArena` calls.
  TfLiteStatus RequestScratchBufferInArena(int node_id, size_t bytes,
                                           int* buffer_idx);
  // Returns the pointer to the planned scratch buffer.
  void* GetScratchBuffer(int buffer_idx) const;

 private:
  TfLiteStatus Init();

  const Model* model_;
  // A simple memory allocator that always allocate from the arena tail.
  SimpleMemoryAllocator* memory_allocator_;
  ErrorReporter* error_reporter_;
  TfLiteContext* context_;
  // Indicating if the allocator is ready for allocation.
  bool active_ = false;

  // In reverse order for efficiency.
  // i.e. scratch_buffer_handles_[0] is the handle for the last buffer,
  // corresponding to the last RequestScratchBufferInArena call.
  internal::ScratchBufferHandle* scratch_buffer_handles_ = nullptr;
  // How many scratch buffers have been allocated.
  size_t scratch_buffer_count_ = 0;

  const SubGraph* subgraph_;
  const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators_;
  const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_MICRO_ALLOCATOR_H_
