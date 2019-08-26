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

#include "tensorflow/lite/experimental/micro/micro_allocator.h"

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/experimental/micro/allocator_utils.h"

namespace tflite {

const int kNeverDestroy = -1, kAllocatedBeforeExecution = -1, kLifetimeNotComputed = -2;

MicroAllocator::MicroAllocator(TfLiteContext* context, const Model* model,
                               uint8_t* tensor_arena, size_t arena_size,
                               ErrorReporter* error_reporter)
    : model_(model),
      tensor_allocator_(context, tensor_arena, arena_size),
      error_reporter_(error_reporter),
      context_(context) {
  auto* subgraphs = model->subgraphs();
  if (subgraphs->size() != 1) {
    error_reporter->Report("Only 1 subgraph is currently supported.\n");
    return;
  }
  subgraph_ = (*subgraphs)[0];
  tensors_ = subgraph_->tensors();
  operators_ = subgraph_->operators();

  context_->tensors_size = tensors_->size();

  uint8_t *allocated_memory;
  tensor_allocator_.AllocateStaticMemory(sizeof(TfLiteTensor) * context_->tensors_size, 4,
          error_reporter_, &allocated_memory);
  context_->tensors = reinterpret_cast<TfLiteTensor*>(allocated_memory);

  tensor_allocator_.AllocateStaticMemory(sizeof(int) * tensors_->size(), sizeof(int),
          error_reporter_, &allocated_memory);
  create_tensor_before_ = reinterpret_cast<int*>(allocated_memory);

  tensor_allocator_.AllocateStaticMemory(sizeof(int) * tensors_->size(), sizeof(int),
                                         error_reporter_, &allocated_memory);
  destroy_tensor_after_ = reinterpret_cast<int*>(allocated_memory);

  // Null all inputs so we can later perform a null check to avoid re-allocating
  // registered pre-allocated inputs.
  for (size_t i = 0; i < subgraph_->inputs()->size(); ++i) {
    const int tensor_index = subgraph_->inputs()->Get(i);
    context_->tensors[tensor_index].data.raw = nullptr;
  }
}

TfLiteStatus MicroAllocator::RegisterPreallocatedInput(uint8_t* buffer,
                                                       size_t input_index) {
  if (buffer == nullptr || input_index < 0 ||
      input_index >= subgraph_->inputs()->size()) {
    error_reporter_->Report("Invalid pre-allocated input %d provided.",
                            input_index);
    return kTfLiteError;
  }
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
      model_->buffers();

  const int tensor_index = subgraph_->inputs()->Get(input_index);
  const auto* tensor = tensors_->Get(tensor_index);
  return this->InitialiseTensor(*tensor, buffers, error_reporter_,
          &context_->tensors[tensor_index], buffer);
}

TfLiteStatus MicroAllocator::InitialiseTensor(
        const tflite::Tensor& flatbuffer_tensor,
        const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
        ErrorReporter* error_reporter, TfLiteTensor* result,
        uint8_t* preallocated_memory) {
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &result->type, error_reporter));
  result->is_variable = flatbuffer_tensor.is_variable();

  result->data.raw = nullptr;
  result->bytes = 0;
  if (auto* buffer = (*buffers)[flatbuffer_tensor.buffer()]) {
    if (auto* array = buffer->data()) {
      if (size_t array_size = array->size()) {
        result->data.raw =
                const_cast<char*>(reinterpret_cast<const char*>(array->data()));
        size_t type_size;
        TF_LITE_ENSURE_STATUS(BytesRequired(flatbuffer_tensor, array_size,
                                            &result->bytes, &type_size,
                                            error_reporter));
      }
    }
  }
  if (result->data.raw) {
    result->allocation_type = kTfLiteMmapRo;
  } else {
    int data_size = 1;
    for (size_t n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
      data_size *= flatbuffer_tensor.shape()->Get(n);
    }
    size_t type_size;
    TF_LITE_ENSURE_STATUS(BytesRequired(flatbuffer_tensor, data_size,
                                        &result->bytes, &type_size,
                                        error_reporter));
    if (preallocated_memory != nullptr) {
      result->data.raw = reinterpret_cast<char*>(preallocated_memory);
    }
    result->allocation_type = kTfLiteArenaRw;
  }

  uint8_t *dims_memory;
  TF_LITE_ENSURE_STATUS(tensor_allocator_.AllocateStaticMemory(
          sizeof(int) * (flatbuffer_tensor.shape()->Length() + 1), sizeof(int), error_reporter, &dims_memory));
  result->dims = reinterpret_cast<TfLiteIntArray*>(dims_memory);

  result->dims->size = flatbuffer_tensor.shape()->Length();
  for (size_t n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
    result->dims->data[n] = flatbuffer_tensor.shape()->Get(n);
  }
  const auto* src_quantization = flatbuffer_tensor.quantization();
  if (src_quantization && src_quantization->scale() &&
      (src_quantization->scale()->size() > 0) &&
      src_quantization->zero_point() &&
      (src_quantization->zero_point()->size() > 0)) {
    result->params.scale = src_quantization->scale()->Get(0);
    for (unsigned int b = 0; b < sizeof(int64_t); ++b)
      *(reinterpret_cast<char*>(&result->params.zero_point) + b) =
              *(reinterpret_cast<const char*>(
                        src_quantization->zero_point()->Data()) +
                b);
    result->params.zero_point =
            flatbuffers::EndianScalar(result->params.zero_point);
  }
  result->allocation = nullptr;
  if (flatbuffer_tensor.name()->c_str() != nullptr) {
    result->name = flatbuffer_tensor.name()->c_str();
  } else {
    result->name = "<No name>";
  }
  result->delegate = nullptr;
  result->buffer_handle = 0;
  result->data_is_stale = false;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::AllocateTensors() {
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
      model_->buffers();

  for (size_t i = 0; i < tensors_->size(); ++i) {
    create_tensor_before_[i] = kLifetimeNotComputed;
    destroy_tensor_after_[i] = kLifetimeNotComputed;
  }

  // It is necessary to specify that model inputs have been allocated to avoid
  // re-allocating later.
  for (size_t i = 0; i < subgraph_->inputs()->size(); ++i) {
    const int tensor_index = subgraph_->inputs()->Get(i);
    const auto* tensor = tensors_->Get(tensor_index);
    // Check for and skip pre-allocated inputs.
    if (context_->tensors[tensor_index].data.raw == nullptr) {
      const TfLiteStatus status = this->InitialiseTensor(
              *tensor, buffers, error_reporter_,
              &context_->tensors[tensor_index]);
      TF_LITE_ENSURE_OK(context_, status);
    }
    // Inputs must be allocated before the first operation runs
    create_tensor_before_[tensor_index] = kAllocatedBeforeExecution;
    destroy_tensor_after_[tensor_index] = kNeverDestroy;
  }

  for (int i = (operators_->size() - 1); i >= 0; --i) {
    const auto* op = operators_->Get(i);

    // Mark all inputs as used by op number `i` at latest
    for (size_t n = 0; n < op->inputs()->size(); ++n) {
      const int tensor_index = op->inputs()->Get(n);
      if (destroy_tensor_after_[tensor_index] == kLifetimeNotComputed) {
        // Since we're iterating backwards through ops, this will get the largest op index
        destroy_tensor_after_[tensor_index] = i;
      }
    }

    for (size_t n = 0; n < op->outputs()->size(); ++n) {
      const int tensor_index = op->outputs()->Get(n);
      if (destroy_tensor_after_[tensor_index] == kLifetimeNotComputed) {
        // By this point, the output of operator `i` would have been consumed by something after it.
        // If it wasn't, it must be an output tensor.
        destroy_tensor_after_[tensor_index] = kNeverDestroy;
      }

      const auto* tensor = tensors_->Get(tensor_index);
      if (!tensor->is_variable()) {
        const TfLiteStatus status =
                this->InitialiseTensor(*tensor, buffers, error_reporter_, &context_->tensors[tensor_index]);
        TF_LITE_ENSURE_OK(context_, status);
        create_tensor_before_[tensor_index] = i;
      }
    }
  }

  // Initialise all variable and read-only tensors
  for (size_t i = 0; i < tensors_->size(); ++i) {
    const auto* tensor = tensors_->Get(i);
    const bool is_read_only = (create_tensor_before_[i] == kLifetimeNotComputed) &&
            (destroy_tensor_after_[i] != kLifetimeNotComputed);
    if (tensor->is_variable() || is_read_only) {
      create_tensor_before_[i] = kAllocatedBeforeExecution;
      destroy_tensor_after_[i] = kNeverDestroy;
      const TfLiteStatus status =
              this->InitialiseTensor(*tensor, buffers, error_reporter_, &context_->tensors[i]);
      TF_LITE_ENSURE_OK(context_, status);
    }

    // Allocate all tensors that have to be allocated before execution
    if (create_tensor_before_[i] == kAllocatedBeforeExecution) {
      this->AllocateTensorBuffer(&context_->tensors[i], error_reporter_);
    }

    // Set default value for variable tensors:
    if (tensor->is_variable()) {
      const TfLiteStatus status = tflite::ResetVariableTensor(&context_->tensors[i]);
      TF_LITE_ENSURE_OK(context_, status);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::AllocateForOperator(int op_index) {
  for (size_t i = 0; i < tensors_->size(); ++i) {
    if (create_tensor_before_[i] == op_index) {
      TF_LITE_ENSURE_STATUS(this->AllocateTensorBuffer(&context_->tensors[i], error_reporter_));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::DeallocateAfterOperator(int op_index) {
  for (size_t i = 0; i < tensors_->size(); ++i) {
    if (destroy_tensor_after_[i] == op_index) {
      TF_LITE_ENSURE_STATUS(this->DeallocateTensorBuffer(&context_->tensors[i], error_reporter_));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::AllocateTensorBuffer(TfLiteTensor* tensor, ErrorReporter* error_reporter) {
  // The pointer is initialised to null when the tensor struct is created.
  // It's up to memory deallocator to set the pointer to `nullptr` if the
  // memory buffer is deleted.
  if (tensor->data.raw != nullptr) {
    return kTfLiteOk;
  }

  tensor_allocator_.AllocateBuffer(tensor, error_reporter);
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::DeallocateTensorBuffer(TfLiteTensor* tensor, ErrorReporter* error_reporter) {
  return tensor_allocator_.DeallocateBuffer(tensor, error_reporter);
}

}  // namespace tflite
