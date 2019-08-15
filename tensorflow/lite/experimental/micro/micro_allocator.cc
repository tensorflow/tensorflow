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

namespace tflite {

MicroAllocator::MicroAllocator(TfLiteContext* context, const Model* model,
                               uint8_t* tensor_arena, size_t arena_size,
                               ErrorReporter* error_reporter)
    : model_(model),
      tensor_allocator_(tensor_arena, arena_size),
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
  context_->tensors =
      reinterpret_cast<TfLiteTensor*>(tensor_allocator_.AllocateMemory(
          sizeof(TfLiteTensor) * context_->tensors_size, 4));

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
  return tensor_allocator_.AllocateTensor(
      *tensor, 0, operators_->size(), buffers, error_reporter_,
      &context_->tensors[tensor_index], buffer);
}

TfLiteStatus MicroAllocator::AllocateTensors() {
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
      model_->buffers();

  int* first_created = reinterpret_cast<int*>(tensor_allocator_.AllocateMemory(
      sizeof(int) * tensors_->size(), sizeof(int)));
  int* last_used = reinterpret_cast<int*>(tensor_allocator_.AllocateMemory(
      sizeof(int) * tensors_->size(), sizeof(int)));
  for (size_t i = 0; i < tensors_->size(); ++i) {
    first_created[i] = -1;
    last_used[i] = -1;
  }

  // It is necessary to specify that model inputs have been allocated to avoid
  // re-allocating later.  Since inputs are not created by a particular node, we
  // make up an index which does not overlap with any node.
  const int kInputIndex = subgraph_->inputs()->size();
  for (size_t i = 0; i < subgraph_->inputs()->size(); ++i) {
    const int tensor_index = subgraph_->inputs()->Get(i);
    const auto* tensor = tensors_->Get(tensor_index);
    // Check for and skip pre-allocated inputs.
    if (context_->tensors[tensor_index].data.raw == nullptr) {
      const TfLiteStatus status = tensor_allocator_.AllocateTensor(
          *tensor, 0, operators_->size(), buffers, error_reporter_,
          &context_->tensors[tensor_index]);
      TF_LITE_ENSURE_OK(context_, status);
    }
    first_created[tensor_index] = kInputIndex;
  }

  for (int i = (operators_->size() - 1); i >= 0; --i) {
    const auto* op = operators_->Get(i);
    for (size_t n = 0; n < op->inputs()->size(); ++n) {
      const int tensor_index = op->inputs()->Get(n);
      if ((last_used[tensor_index] == -1) || (last_used[tensor_index] < i)) {
        last_used[tensor_index] = i;
      }
    }
    for (size_t n = 0; n < op->outputs()->size(); ++n) {
      const int tensor_index = op->outputs()->Get(n);
      const int create_before = i;
      int destroy_after = last_used[tensor_index];
      if (destroy_after == -1) {
        destroy_after = operators_->size();
      }
      const auto* tensor = tensors_->Get(tensor_index);
      if (!tensor->is_variable()) {
        const TfLiteStatus status = tensor_allocator_.AllocateTensor(
            *tensor, create_before, destroy_after, buffers, error_reporter_,
            &context_->tensors[tensor_index]);
        if (status != kTfLiteOk) {
          return status;
        }
        first_created[tensor_index] = i;
      }
    }
  }

  for (size_t i = 0; i < tensors_->size(); ++i) {
    const auto* tensor = tensors_->Get(i);
    const bool is_read_only = (first_created[i] == -1) && (last_used[i] != -1);
    if (tensor->is_variable() || is_read_only) {
      const TfLiteStatus status = tensor_allocator_.AllocateTensor(
          *tensor, 0, operators_->size(), buffers, error_reporter_,
          &context_->tensors[i]);
      if (status != kTfLiteOk) {
        return status;
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace tflite
