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
#include "tensorflow/lite/micro/micro_error_reporter.h"
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

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   ErrorReporter* error_reporter,
                                   MicroProfiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*MicroAllocator::Create(tensor_arena, tensor_arena_size,
                                         error_reporter)),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      eval_tensors_(nullptr),
      input_tensors_(nullptr),
      output_tensors_(nullptr) {
  Init(profiler);
}

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   MicroAllocator* allocator,
                                   ErrorReporter* error_reporter,
                                   MicroProfiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*allocator),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      eval_tensors_(nullptr),
      input_tensors_(nullptr),
      output_tensors_(nullptr) {
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

void MicroInterpreter::Init(MicroProfiler* profiler) {
  const flatbuffers::Vector<flatbuffers::Offset<SubGraph>>* subgraphs =
      model_->subgraphs();
  if (subgraphs->size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Only 1 subgraph is currently supported.\n");
    initialization_status_ = kTfLiteError;
    return;
  }
  subgraph_ = (*subgraphs)[0];

  context_.impl_ = static_cast<void*>(this);
  context_.ReportError = ReportOpError;
  context_.GetTensor = GetTensor;
  context_.GetEvalTensor = GetEvalTensor;
  context_.recommended_num_threads = 1;
  context_.profiler = profiler;

  initialization_status_ = kTfLiteOk;
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

  context_.tensors_size = subgraph_->tensors()->size();

  // Only allow AllocatePersistentBuffer in Init stage.
  context_.AllocatePersistentBuffer = AllocatePersistentBuffer;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = nullptr;

  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
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

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is
  // available in Prepare stage.
  context_.RequestScratchBufferInArena = RequestScratchBufferInArena;
  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
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
    allocator_.FinishPrepareNodeAllocations(/*node_id=*/i);
  }

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  context_.AllocatePersistentBuffer = nullptr;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = GetScratchBuffer;

  TF_LITE_ENSURE_OK(&context_,
                    allocator_.FinishModelAllocation(model_, eval_tensors_,
                                                     &scratch_buffer_handles_));

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  input_tensors_ =
      reinterpret_cast<TfLiteTensor**>(allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor*) * inputs_size()));
  if (input_tensors_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory for context->input_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor*) * inputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < inputs_size(); ++i) {
    input_tensors_[i] = allocator_.AllocatePersistentTfLiteTensor(
        model_, eval_tensors_, inputs().Get(i));
    if (input_tensors_[i] == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Failed to initialize input tensor %d", i);
      return kTfLiteError;
    }
  }

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  output_tensors_ =
      reinterpret_cast<TfLiteTensor**>(allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor*) * outputs_size()));
  if (output_tensors_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory for context->output_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor*) * outputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < outputs_size(); ++i) {
    output_tensors_[i] = allocator_.AllocatePersistentTfLiteTensor(
        model_, eval_tensors_, outputs().Get(i));
    if (output_tensors_[i] == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Failed to initialize output tensor %d", i);
      return kTfLiteError;
    }
  }

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

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfiler*>(context_.profiler));
#endif

    TFLITE_DCHECK(registration->invoke);
    TfLiteStatus invoke_status = registration->invoke(&context_, node);

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
  return input_tensors_[index];
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const size_t length = outputs_size();
  if (index >= length) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Output index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return output_tensors_[index];
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

void* MicroInterpreter::AllocatePersistentBuffer(TfLiteContext* context,
                                                 size_t bytes) {
  return reinterpret_cast<MicroInterpreter*>(context->impl_)
      ->allocator_.AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroInterpreter::RequestScratchBufferInArena(
    TfLiteContext* context, size_t bytes, int* buffer_idx) {
  // All scratch buffer requests are managed in the allocator. Simply route the
  // request and let the allocator manage allocations.
  return static_cast<MicroInterpreter*>(context->impl_)
      ->allocator_.RequestScratchBufferInArena(bytes, buffer_idx);
}

void* MicroInterpreter::GetScratchBuffer(TfLiteContext* context,
                                         int buffer_idx) {
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  ScratchBufferHandle* handle =
      interpreter->scratch_buffer_handles_ + buffer_idx;
  return handle->data;
}

void MicroInterpreter::ReportOpError(struct TfLiteContext* context,
                                     const char* format, ...) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(interpreter->error_reporter_, format, args);
  va_end(args);
#endif
}

TfLiteTensor* MicroInterpreter::GetTensor(const struct TfLiteContext* context,
                                          int tensor_idx) {
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  return interpreter->allocator_.AllocateTempTfLiteTensor(
      interpreter->model_, interpreter->eval_tensors_, tensor_idx);
}

TfLiteEvalTensor* MicroInterpreter::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_idx) {
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  return &interpreter->eval_tensors_[tensor_idx];
}

}  // namespace tflite
