/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

#ifndef TF_LITE_STRIP_ERROR_STRINGS
const char* OpNameFromRegistration(const TfLiteRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}
#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)

}  // namespace

namespace internal {

ContextHelper::ContextHelper(ErrorReporter* error_reporter,
                             MicroAllocator* allocator, const Model* model)
    : allocator_(allocator), error_reporter_(error_reporter), model_(model) {}

void* ContextHelper::AllocatePersistentBuffer(TfLiteContext* ctx,
                                              size_t bytes) {
  return reinterpret_cast<ContextHelper*>(ctx->impl_)
      ->allocator_->AllocatePersistentBuffer(bytes);
}

TfLiteStatus ContextHelper::RequestScratchBufferInArena(TfLiteContext* ctx,
                                                        size_t bytes,
                                                        int* buffer_idx) {
  ContextHelper* helper = reinterpret_cast<ContextHelper*>(ctx->impl_);

  // We can not forward the scratch buffer request to the allocator yet,
  // otherwise the scratch buffer handles will ruin the data in `temp` section.
  // These requests will be processed once the `temp` section is deallocated,
  // i.e. after a node has been prepared.

  if (helper->scratch_buffer_count_ >= kMaxScratchBuffersPerOp) {
    TF_LITE_REPORT_ERROR(
        helper->error_reporter_,
        "Node %d is allocating too many scratch buffers per op, max=%d",
        helper->current_node_idx_, helper->scratch_buffer_count_);
  }
  helper->scrach_buffer_sizes_[helper->scratch_buffer_count_] = bytes;
  // buffer_idx is 0 indexed.
  *buffer_idx = helper->scratch_buffer_count_ +
                helper->allocator_->GetScratchBufferCount();
  helper->scratch_buffer_count_++;
  return kTfLiteOk;
}

void* ContextHelper::GetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  ContextHelper* helper = reinterpret_cast<ContextHelper*>(ctx->impl_);

  return helper->allocator_->GetScratchBuffer(helper->scratch_buffer_handles_,
                                              buffer_idx);
}

void ContextHelper::ReportOpError(struct TfLiteContext* context,
                                  const char* format, ...) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  ContextHelper* helper = static_cast<ContextHelper*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(helper->error_reporter_, format, args);
  va_end(args);
#endif
}

TfLiteTensor* ContextHelper::GetTensor(const struct TfLiteContext* context,
                                       int tensor_idx) {
  ContextHelper* helper = static_cast<ContextHelper*>(context->impl_);
  return helper->allocator_->AllocateTempTfLiteTensor(
      helper->model_, helper->eval_tensors_, tensor_idx);
}

TfLiteEvalTensor* ContextHelper::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_idx) {
  ContextHelper* helper = reinterpret_cast<ContextHelper*>(context->impl_);
  return &helper->eval_tensors_[tensor_idx];
}

void ContextHelper::SetNodeIndex(int idx) {
  if (scratch_buffer_count_ != 0) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Internal error: Please commit scratch buffers "
                         "befrore moving to the next node");
  }
  current_node_idx_ = idx;
}

void ContextHelper::SetTfLiteEvalTensors(TfLiteEvalTensor* eval_tensors) {
  eval_tensors_ = eval_tensors;
}

void ContextHelper::SetScratchBufferHandles(void* scratch_buffer_handle) {
  scratch_buffer_handles_ = scratch_buffer_handle;
}

TfLiteStatus ContextHelper::CommitScratchBuffers() {
  size_t initial_buffer_count = allocator_->GetScratchBufferCount();
  for (size_t i = 0; i < scratch_buffer_count_; i++) {
    int buffer_id;
    allocator_->RequestScratchBufferInArena(
        current_node_idx_, scrach_buffer_sizes_[i], &buffer_id);
    if (static_cast<size_t>(buffer_id) != initial_buffer_count + i) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Internal error. Scratch buffers are not contiguous.\n");
    }
  }
  scratch_buffer_count_ = 0;
  return kTfLiteOk;
}

}  // namespace internal

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   ErrorReporter* error_reporter,
                                   tflite::Profiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*MicroAllocator::Create(tensor_arena, tensor_arena_size,
                                         error_reporter)),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      eval_tensors_(nullptr),
      context_helper_(error_reporter_, &allocator_, model),
      input_tensor_(nullptr),
      output_tensor_(nullptr) {
  Init(profiler);
}

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   MicroAllocator* allocator,
                                   ErrorReporter* error_reporter,
                                   tflite::Profiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*allocator),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      eval_tensors_(nullptr),
      context_helper_(error_reporter_, &allocator_, model),
      input_tensor_(nullptr),
      output_tensor_(nullptr) {
  Init(profiler);
}

MicroInterpreter::~MicroInterpreter() {
  if (node_and_registrations_ != nullptr) {
    for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
      TfLiteNode* node = &(node_and_registrations_[i].node);
      const TfLiteRegistration* registration =
          node_and_registrations_[i].registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(&context_, node->user_data);
      }
    }
  }
}

void MicroInterpreter::Init(tflite::Profiler* profiler) {
  const flatbuffers::Vector<flatbuffers::Offset<SubGraph>>* subgraphs =
      model_->subgraphs();
  if (subgraphs->size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Only 1 subgraph is currently supported.\n");
    initialization_status_ = kTfLiteError;
    return;
  }
  subgraph_ = (*subgraphs)[0];

  context_.impl_ = static_cast<void*>(&context_helper_);
  context_.ReportError = context_helper_.ReportOpError;
  context_.GetTensor = context_helper_.GetTensor;
  context_.GetEvalTensor = context_helper_.GetEvalTensor;
  context_.recommended_num_threads = 1;
  context_.profiler = profiler;

  initialization_status_ = kTfLiteOk;
}

void MicroInterpreter::CorrectTensorEndianness(TfLiteEvalTensor* tensorCorr) {
  int32_t tensorSize = 1;
  for (int d = 0; d < tensorCorr->dims->size; ++d)
    tensorSize *= reinterpret_cast<const int32_t*>(tensorCorr->dims->data)[d];

  switch (tensorCorr->type) {
    case TfLiteType::kTfLiteFloat32:
      CorrectTensorDataEndianness(tensorCorr->data.f, tensorSize);
      break;
    case TfLiteType::kTfLiteFloat16:
      CorrectTensorDataEndianness(tensorCorr->data.f16, tensorSize);
      break;
    case TfLiteType::kTfLiteInt64:
      CorrectTensorDataEndianness(tensorCorr->data.i64, tensorSize);
      break;
    case TfLiteType::kTfLiteInt32:
      CorrectTensorDataEndianness(tensorCorr->data.i32, tensorSize);
      break;
    case TfLiteType::kTfLiteInt16:
      CorrectTensorDataEndianness(tensorCorr->data.i16, tensorSize);
      break;
    case TfLiteType::kTfLiteComplex64:
      CorrectTensorDataEndianness(tensorCorr->data.c64, tensorSize);
      break;
    case TfLiteType::kTfLiteComplex128:
      CorrectTensorDataEndianness(tensorCorr->data.c128, tensorSize);
      break;
    default:
      // Do nothing for other data types.
      break;
  }
}

template <class T>
void MicroInterpreter::CorrectTensorDataEndianness(T* data, int32_t size) {
  for (int32_t i = 0; i < size; ++i) {
    data[i] = flatbuffers::EndianScalar(data[i]);
  }
}

TfLiteStatus MicroInterpreter::AllocateTensors() {
  if (allocator_.StartModelAllocation(model_, op_resolver_,
                                      &node_and_registrations_,
                                      &eval_tensors_) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed starting model allocation.\n");
    initialization_status_ = kTfLiteError;
    return kTfLiteError;
  }

  // Update the pointer now that TfLiteEvalTensor allocation has completed on
  // the context helper.
  // TODO(b/16157777): This call would not be needed if ContextHelper rolled
  // into the interpreter.
  context_helper_.SetTfLiteEvalTensors(eval_tensors_);
  context_.tensors_size = subgraph_->tensors()->size();

  // If the system is big endian then convert weights from the flatbuffer from
  // little to big endian on startup so that it does not need to be done during
  // inference.
  // NOTE: This requires that the flatbuffer is held in memory which can be
  // modified by this process.
  if (!FLATBUFFERS_LITTLEENDIAN) {
    for (size_t t = 0; t < subgraph_->tensors()->size(); ++t) {
      if (auto* buffer =
              (*model_->buffers())[subgraph_->tensors()->Get(t)->buffer()]) {
        // If we've found a buffer, does it have any data?
        if (auto* array = buffer->data()) {
          // If it has any data, is the data size larger than zero?
          if (array->size()) {
            // Update the endianness of the corresponding eval tensor since that
            // struct holds the buffer used at inference time.
            CorrectTensorEndianness(&eval_tensors_[t]);
          }
        }
      }
    }
  }

  // Only allow AllocatePersistentBuffer in Init stage.
  context_.AllocatePersistentBuffer = context_helper_.AllocatePersistentBuffer;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = nullptr;

  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    context_helper_.SetNodeIndex(i);
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;
    size_t init_data_size;
    const char* init_data;
    if (registration->builtin_code == BuiltinOperator_CUSTOM) {
      init_data = reinterpret_cast<const char*>(node->custom_initial_data);
      init_data_size = node->custom_initial_data_size;
    } else {
      init_data = reinterpret_cast<const char*>(node->builtin_data);
      init_data_size = 0;
    }
    if (registration->init) {
      node->user_data =
          registration->init(&context_, init_data, init_data_size);
    }
  }
  context_helper_.SetNodeIndex(-1);

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is
  // available in Prepare stage.
  context_.RequestScratchBufferInArena =
      context_helper_.RequestScratchBufferInArena;
  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    // Set node idx to annotate the lifetime for scratch buffers.
    context_helper_.SetNodeIndex(i);
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;
    if (registration->prepare) {
      TfLiteStatus prepare_status = registration->prepare(&context_, node);
      if (prepare_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Node %s (number %df) failed to prepare with status %d",
            OpNameFromRegistration(registration), i, prepare_status);
        return kTfLiteError;
      }
    }
    allocator_.ResetTempAllocations();
    context_helper_.CommitScratchBuffers();
  }
  context_helper_.SetNodeIndex(-1);

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  context_.AllocatePersistentBuffer = nullptr;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = context_helper_.GetScratchBuffer;

  void* scratch_buffer_handles = nullptr;

  TF_LITE_ENSURE_OK(&context_,
                    allocator_.FinishModelAllocation(model_, eval_tensors_,
                                                     &scratch_buffer_handles));
  context_helper_.SetScratchBufferHandles(scratch_buffer_handles);
  TF_LITE_ENSURE_STATUS(ResetVariableTensors());

  tensors_allocated_ = true;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::Invoke() {
  if (initialization_status_ != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Invoke() called after initialization failed\n");
    return kTfLiteError;
  }

  // Ensure tensors are allocated before the interpreter is invoked to avoid
  // difficult to debug segfaults.
  if (!tensors_allocated_) {
    TF_LITE_ENSURE_OK(&context_, AllocateTensors());
  }

  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;

    if (registration->invoke) {
      TfLiteStatus invoke_status;
#ifndef NDEBUG  // Omit profiler overhead from release builds.
      // The case where profiler == nullptr is handled by
      // ScopedOperatorProfile.
      tflite::Profiler* profiler =
          reinterpret_cast<tflite::Profiler*>(context_.profiler);
      ScopedOperatorProfile scoped_profiler(
          profiler, OpNameFromRegistration(registration), i);
#endif
      invoke_status = registration->invoke(&context_, node);

      // All TfLiteTensor structs used in the kernel are allocated from temp
      // memory in the allocator. This creates a chain of allocations in the
      // temp section. The call below resets the chain of allocations to
      // prepare for the next call.
      allocator_.ResetTempAllocations();

      if (invoke_status == kTfLiteError) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Node %s (number %d) failed to invoke with status %d",
            OpNameFromRegistration(registration), i, invoke_status);
        return kTfLiteError;
      } else if (invoke_status != kTfLiteOk) {
        return invoke_status;
      }
    }
  }
  return kTfLiteOk;
}

TfLiteTensor* MicroInterpreter::input(size_t index) {
  const size_t length = inputs_size();
  if (index >= length) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Input index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  if (index != 0) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Input tensors not at index 0 are allocated from the "
        "persistent memory arena. Repeat calls will cause excess "
        "allocation!");
    return allocator_.AllocatePersistentTfLiteTensor(model_, eval_tensors_,
                                                     inputs().Get(index));
  }
  if (input_tensor_ == nullptr) {
    input_tensor_ = allocator_.AllocatePersistentTfLiteTensor(
        model_, eval_tensors_, inputs().Get(index));
  }
  return input_tensor_;
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const size_t length = outputs_size();
  if (index >= length) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Output index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  if (index != 0) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Output tensors not at index 0 are allocated from the "
        "persistent memory arena. Repeat calls will cause excess "
        "allocation!");
    return allocator_.AllocatePersistentTfLiteTensor(model_, eval_tensors_,
                                                     outputs().Get(index));
  }
  if (output_tensor_ == nullptr) {
    // TODO(b/160894903): This API will allocate TfLiteTensor structs from
    // persistent (tail) memory and cache on this pointer.
    output_tensor_ = allocator_.AllocatePersistentTfLiteTensor(
        model_, eval_tensors_, outputs().Get(index));
  }
  return output_tensor_;
}

TfLiteTensor* MicroInterpreter::tensor(size_t index) {
  const size_t length = tensors_size();
  if (index >= length) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Tensor index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return allocator_.AllocatePersistentTfLiteTensor(model_, eval_tensors_,
                                                   index);
}

TfLiteStatus MicroInterpreter::ResetVariableTensors() {
  for (size_t i = 0; i < subgraph_->tensors()->size(); ++i) {
    auto* tensor = subgraph_->tensors()->Get(i);
    if (tensor->is_variable()) {
      size_t buffer_size;
      TF_LITE_ENSURE_STATUS(
          TfLiteEvalTensorByteLength(&eval_tensors_[i], &buffer_size));

      int value = 0;
      if (tensor->type() == tflite::TensorType_INT8) {
        value = tensor->quantization()->zero_point()->Get(0);
      }
      memset(eval_tensors_[i].data.raw, value, buffer_size);
    }
  }

  return kTfLiteOk;
}

}  // namespace tflite
