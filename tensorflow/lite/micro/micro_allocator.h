/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
b/160894903
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

#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Namespace used for unittests.
namespace internal {

// Sets up all of the data structure members for a TfLiteTensor based on the
// contents of a serialized tensor in the flatbuffer.
// TODO(b/160894903): Once all kernels have been updated to the new
// TfLiteEvalTensor API - drop the allocate_temp flag. This enables internal
// flatbuffer quantization or dimension allocations to take place in either the
// temp or tail section of the arena.
TfLiteStatus InitializeTfLiteTensorFromFlatbuffer(
    SimpleMemoryAllocator* allocator, bool allocate_temp,
    const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result);

// A handle tracking scratch buffer allocation. This handle is created by
// `RequestScratchBufferInArena`. `data` field is populated in
// `FinishModelAllocation` after static memory planning.
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
//
// The lifetime of the model, tensor arena and error reporter must be at
// least as long as that of the allocator object, since the allocator needs
// them to be accessible during its entire lifetime.
//
// The MicroAllocator simply plans out additional allocations that are required
// to standup a model for inference in TF Micro. This class currently relies on
// an additional allocator - SimpleMemoryAllocator - for all allocations from an
// arena. These allocations are divided into head (non-persistent) and tail
// (persistent) regions:
//
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
  // Creates a MicroAllocator instance from a given tensor arena. This arena
  // will be managed by the created instance.
  // Note: Please use __declspec(align(16)) to make sure tensor_arena is 16
  // bytes aligned, otherwise some head room will be wasted.
  // TODO(b/157615197): Cleanup constructor + factory usage.
  static MicroAllocator* Create(uint8_t* tensor_arena, size_t arena_size,
                                ErrorReporter* error_reporter);

  // Creates a MicroAllocator instance using the provided SimpleMemoryAllocator
  // intance. This allocator instance will use the SimpleMemoryAllocator
  // instance to manage allocations internally.
  static MicroAllocator* Create(SimpleMemoryAllocator* memory_allocator,
                                ErrorReporter* error_reporter);

  // Begin allocating internal resources required for model inference.
  // This method will run through the flatbuffer data supplied in the model to
  // properly allocate tensor, node, and op registration data. This method is
  // expected to be followed with a call to FinishModelAllocation() before
  // resuming allocation with another model. All persistent tensor buffers are
  // stored in the out-param eval_tensors. This value is allocated from the
  // persistent memory arena and will be used to host runtime tensor buffers.
  TfLiteStatus StartModelAllocation(
      const Model* model, const MicroOpResolver& op_resolver,
      NodeAndRegistration** node_and_registrations,
      TfLiteEvalTensor** eval_tensors);

  // Finish allocating internal resources required for model inference.
  // This method will plan non-persistent buffers and commit a memory plan to
  // the 'head' section of the memory arena. All variable tensor data will also
  // be allocated. This method should be called after assigning model resources
  // in StartModelAllocation(). The eval_tensors pointer should be the value
  // passed into this class during StartModelAllocation(). Scratch buffer
  // handles are stored in the out-param `scratch_buffer_handles`. This value
  // will be used in `GetScratchBuffer` call to retrieve scratch buffers.
  TfLiteStatus FinishModelAllocation(const Model* model,
                                     TfLiteEvalTensor* eval_tensors,
                                     void** scratch_buffer_handles = nullptr);

  // Allocates a TfLiteTensor struct and populates the returned value with
  // properties from the model flatbuffer. This struct is allocated from
  // persistent arena memory is only guaranteed for the lifetime of the
  // application. The eval_tensors pointer should be the value passed into this
  // class during StartModelAllocation() and contains the source-of-truth for
  // buffers.
  virtual TfLiteTensor* AllocatePersistentTfLiteTensor(
      const Model* model, TfLiteEvalTensor* eval_tensors, int tensor_index);

  // Allocates a TfLiteTensor struct and populates the returned value with
  // properties from the model flatbuffer. This struct is allocated from
  // temporary arena memory is only guaranteed until a call is made to
  // ResetTempAllocations(). The eval_tensors pointer should be the value passed
  // into this class during StartModelAllocation() and contains the
  // source-of-truth for buffers.
  virtual TfLiteTensor* AllocateTempTfLiteTensor(const Model* model,
                                                 TfLiteEvalTensor* eval_tensors,
                                                 int tensor_index);

  // Resets all temporary allocations. This method should be called after a
  // chain of temp allocations (e.g. chain of TfLiteTensor objects via
  // AllocateTfLiteTensor()).
  virtual void ResetTempAllocations();

  // Allocates persistent buffer which has the same life time as the allocator.
  // The memory is immediately available and is allocated from the tail of the
  // arena.
  void* AllocatePersistentBuffer(size_t bytes);

  // Register a scratch buffer of size `bytes` for Node with `node_id`.
  // This method only allocates a BufferHandle holding information for memory
  // planning. The buffer ptr is ready after `FinishModelAllocation` and can
  // be retrieved by `GetScratchBuffer` method using the returned buffer_idx.
  // Note that this method should only be called in the Prepare stage.
  TfLiteStatus RequestScratchBufferInArena(int node_id, size_t bytes,
                                           int* buffer_idx);

  // Return the number of scratch buffers in the allocator.
  size_t GetScratchBufferCount() const { return scratch_buffer_count_; }

  // Return the pointer to the planned scratch buffer. `scratch_buffer_handles`
  // should be the corresponding value returned in `FinishModelAllocation`.
  // `scratch_buffer_handles` is intentionally desigend as void*. The actual
  // data type is an implementation detail, and is only visible in this class.
  static void* GetScratchBuffer(void* scratch_buffer_handles, int buffer_idx);

  // Returns the arena usage in bytes, only available after
  // `FinishModelAllocation`. Otherwise, it will return 0.
  size_t used_bytes() const;

 protected:
  MicroAllocator(SimpleMemoryAllocator* memory_allocator,
                 ErrorReporter* error_reporter);
  virtual ~MicroAllocator();

  // Allocates an array in the arena to hold pointers to the node and
  // registration pointers required to represent the inference graph of the
  // model.
  virtual TfLiteStatus AllocateNodeAndRegistrations(
      const Model* model, NodeAndRegistration** node_and_registrations);

  // Populates node and registration pointers representing the inference graph
  // of the model from values inside the flatbuffer (loaded from the TfLiteModel
  // instance). Persistent data (e.g. operator data) is allocated from the
  // arena.
  virtual TfLiteStatus PrepareNodeAndRegistrationDataFromFlatbuffer(
      const Model* model, const MicroOpResolver& op_resolver,
      NodeAndRegistration* node_and_registrations);

  // Allocates the list of persistent TfLiteEvalTensors that are used for the
  // "eval" phase of model inference. These structs will be the source of truth
  // for all tensor buffers. Allocation results are stored in the out-param
  // eval_tensors.
  virtual TfLiteStatus AllocateTfLiteEvalTensors(
      const Model* model, TfLiteEvalTensor** eval_tensors);

  // Allocates persistent tensor buffers for variable tensors in the subgraph.
  virtual TfLiteStatus AllocateVariables(const SubGraph* subgraph,
                                         TfLiteEvalTensor* eval_tensors);

  // TODO(b/160894903): Once all kernels have been updated to the new API drop
  // this method. It is only used to record TfLiteTensor persistent allocations.
  virtual TfLiteTensor* AllocatePersistentTfLiteTensorInternal(
      const Model* model, TfLiteEvalTensor* eval_tensors, int tensor_index);

  // Populates a TfLiteTensor struct with data from the model flatbuffer. Any
  // quantization data is allocated from either the tail (persistent) or temp
  // sections of the arena based on the allocation flag.
  // TODO(b/160894903): Once all kernels have been updated to the new API drop
  // this function since all allocations for quantized data will take place in
  // the temp section.
  virtual TfLiteStatus PopulateTfLiteTensorFromFlatbuffer(
      const Model* model, const SubGraph* subgraph, TfLiteTensor* tensor,
      int tensor_index, bool allocate_temp);

  ErrorReporter* error_reporter() const;

  // Returns the first subgraph from the model.
  const SubGraph* GetSubGraphFromModel(const Model* model);

 private:
  // Commits a memory plan for all non-persistent buffer allocations in the
  // 'head' section of the memory arena. The eval_tensors pointer is the list of
  // pre-allocated TfLiteEvalTensor structs that will point to the buffers that
  // will be allocated into the head section in this function call.
  virtual TfLiteStatus CommitStaticMemoryPlan(const Model* model,
                                              const SubGraph* subgraph,
                                              TfLiteEvalTensor* eval_tensors);

  // A simple memory allocator that always allocate from the arena tail or head.
  SimpleMemoryAllocator* memory_allocator_;

  ErrorReporter* error_reporter_;
  bool model_is_allocating_;

  // Points to the first allocated scratch buffer handle.
  // Scratch buffer handles are placed in the head during `Prepare` stage and
  // then moved to the tail for static memory plan.
  internal::ScratchBufferHandle* scratch_buffer_handles_ = nullptr;
  // How many scratch buffers have been allocated.
  size_t scratch_buffer_count_ = 0;

  virtual TfLiteStatus InitScratchBufferHandles();
  virtual TfLiteStatus MoveScratchBufferHandlesToTail();

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_MICRO_ALLOCATOR_H_
