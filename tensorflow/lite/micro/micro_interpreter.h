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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_

#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Copied from tensorflow/lite/version.h to avoid a dependency chain into
// tensorflow/core.
#define TFLITE_SCHEMA_VERSION (3)

namespace tflite {

namespace internal {

// A helper class to encapsulate the implementation of APIs in Context.
// context->impl_ points to an instance of this class.
// Check tensorflow/lite/c/common.h for detailed descriptions.
// TODO(b/16157777): Consider rolling this class into MicroInterpreter.
class ContextHelper {
 public:
  explicit ContextHelper(ErrorReporter* error_reporter,
                         MicroAllocator* allocator, const Model* model);

  // Functions that will be assigned to function pointers on TfLiteContext:
  static void* AllocatePersistentBuffer(TfLiteContext* ctx, size_t bytes);
  static TfLiteStatus RequestScratchBufferInArena(TfLiteContext* ctx,
                                                  size_t bytes,
                                                  int* buffer_idx);
  static void* GetScratchBuffer(TfLiteContext* ctx, int buffer_idx);
  static void ReportOpError(struct TfLiteContext* context, const char* format,
                            ...);
  static TfLiteTensor* GetTensor(const struct TfLiteContext* context,
                                 int tensor_idx);
  static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                         int tensor_idx);

  // Sets the pointer to a list of TfLiteEvalTensor instances.
  void SetTfLiteEvalTensors(TfLiteEvalTensor* eval_tensors);

  // Sets the pointer to a list of ScratchBufferHandle instances.
  void SetScratchBufferHandles(ScratchBufferHandle* scratch_buffer_handles);

 private:
  MicroAllocator* allocator_ = nullptr;
  ErrorReporter* error_reporter_ = nullptr;
  const Model* model_ = nullptr;
  TfLiteEvalTensor* eval_tensors_ = nullptr;
  ScratchBufferHandle* scratch_buffer_handles_ = nullptr;
};

}  // namespace internal

class MicroInterpreter {
 public:
  // The lifetime of the model, op resolver, tensor arena, error reporter and
  // profiler must be at least as long as that of the interpreter object, since
  // the interpreter may need to access them at any time. This means that you
  // should usually create them with the same scope as each other, for example
  // having them all allocated on the stack as local variables through a
  // top-level function. The interpreter doesn't do any deallocation of any of
  // the pointed-to objects, ownership remains with the caller.
  MicroInterpreter(const Model* model, const MicroOpResolver& op_resolver,
                   uint8_t* tensor_arena, size_t tensor_arena_size,
                   ErrorReporter* error_reporter,
                   MicroProfiler* profiler = nullptr);

  // Create an interpreter instance using an existing MicroAllocator instance.
  // This constructor should be used when creating an allocator that needs to
  // have allocation handled in more than one interpreter or for recording
  // allocations inside the interpreter. The lifetime of the allocator must be
  // as long as that of the interpreter object.
  MicroInterpreter(const Model* model, const MicroOpResolver& op_resolver,
                   MicroAllocator* allocator, ErrorReporter* error_reporter,
                   MicroProfiler* profiler = nullptr);

  ~MicroInterpreter();

  // Runs through the model and allocates all necessary input, output and
  // intermediate tensors.
  TfLiteStatus AllocateTensors();

  // In order to support partial graph runs for strided models, this can return
  // values other than kTfLiteOk and kTfLiteError.
  // TODO(b/149795762): Add this to the TfLiteStatus enum.
  TfLiteStatus Invoke();

  size_t tensors_size() const { return context_.tensors_size; }
  TfLiteTensor* tensor(size_t tensor_index);
  template <class T>
  T* typed_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

  TfLiteTensor* input(size_t index);
  size_t inputs_size() const { return subgraph_->inputs()->Length(); }
  const flatbuffers::Vector<int32_t>& inputs() const {
    return *subgraph_->inputs();
  }
  TfLiteTensor* input_tensor(size_t index) { return input(index); }
  template <class T>
  T* typed_input_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = input_tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

  TfLiteTensor* output(size_t index);
  size_t outputs_size() const { return subgraph_->outputs()->Length(); }
  const flatbuffers::Vector<int32_t>& outputs() const {
    return *subgraph_->outputs();
  }
  TfLiteTensor* output_tensor(size_t index) { return output(index); }
  template <class T>
  T* typed_output_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = output_tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

  // Reset all variable tensors to the default value.
  TfLiteStatus ResetVariableTensors();

  TfLiteStatus initialization_status() const { return initialization_status_; }

  size_t operators_size() const { return subgraph_->operators()->size(); }

  // For debugging only.
  const NodeAndRegistration node_and_registration(int node_index) const {
    return node_and_registrations_[node_index];
  }

  // For debugging only.
  // Returns the actual used arena in bytes. This method gives the optimal arena
  // size. It's only available after `AllocateTensors` has been called.
  // Note that normally `tensor_arena` requires 16 bytes alignment to fully
  // utilize the space. If it's not the case, the optimial arena size would be
  // arena_used_bytes() + 16.
  size_t arena_used_bytes() const { return allocator_.used_bytes(); }

 protected:
  const MicroAllocator& allocator() const { return allocator_; }
  const TfLiteContext& context() const { return context_; }

 private:
  // TODO(b/158263161): Consider switching to Create() function to enable better
  // error reporting during initialization.
  void Init(MicroProfiler* profiler);

  NodeAndRegistration* node_and_registrations_ = nullptr;

  const Model* model_;
  const MicroOpResolver& op_resolver_;
  ErrorReporter* error_reporter_;
  TfLiteContext context_ = {};
  MicroAllocator& allocator_;
  bool tensors_allocated_;

  TfLiteStatus initialization_status_;

  const SubGraph* subgraph_ = nullptr;
  TfLiteEvalTensor* eval_tensors_ = nullptr;
  ScratchBufferHandle* scratch_buffer_handles_ = nullptr;

  // TODO(b/16157777): Drop this reference:
  internal::ContextHelper context_helper_;

  // TODO(b/162311891): Clean these pointers up when this class supports buffers
  // from TfLiteEvalTensor.
  TfLiteTensor** input_tensors_;
  TfLiteTensor** output_tensors_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_
