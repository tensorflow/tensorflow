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
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/experimental/micro/memory_helpers.h"
#include "tensorflow/lite/experimental/micro/memory_planner/greedy_memory_planner.h"

namespace tflite {

namespace {
// Used to hold information used during allocation calculations.
struct TensorInfo {
  const tflite::Tensor* flatbuffer_tensor;
  TfLiteTensor* runtime_tensor;
  int first_created;
  int last_used;
  bool needs_allocating;
};

// We align tensor buffers to 16-byte boundaries, since this is a common
// requirement for SIMD extensions.
constexpr int kBufferAlignment = 16;

}  // namespace

MicroAllocator::MicroAllocator(TfLiteContext* context, const Model* model,
                               uint8_t* tensor_arena, size_t arena_size,
                               ErrorReporter* error_reporter)
    : model_(model),
      memory_allocator_(tensor_arena, arena_size),
      error_reporter_(error_reporter),
      context_(context),
      arena_(tensor_arena),
      arena_size_(arena_size) {
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
      reinterpret_cast<TfLiteTensor*>(memory_allocator_.AllocateFromTail(
          sizeof(TfLiteTensor) * context_->tensors_size, 4));

  // Null all inputs so we can later perform a null check to avoid re-allocating
  // registered pre-allocated inputs.
  for (size_t i = 0; i < context_->tensors_size; ++i) {
    context_->tensors[i].data.raw = nullptr;
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
  return InitializeRuntimeTensor(*tensor, buffers, error_reporter_,
                                 &context_->tensors[tensor_index], buffer);
}

TfLiteStatus MicroAllocator::AllocateTensors() {
  const size_t tensors_size = tensors_->size();

  // It would be better not to allocate this memory for the lifetime of the
  // model, but we don't have a straightforward way to avoid it.
  TensorInfo* tensor_info =
      reinterpret_cast<TensorInfo*>(memory_allocator_.AllocateFromTail(
          sizeof(TensorInfo) * tensors_size, sizeof(TensorInfo)));

  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
      model_->buffers();

  // Set up the runtime data structures for all tensors.
  for (size_t i = 0; i < tensors_size; ++i) {
    TensorInfo* current = &tensor_info[i];
    current->flatbuffer_tensor = &(*(tensors_->Get(i)));
    current->runtime_tensor = &context_->tensors[i];
    const bool is_variable = current->flatbuffer_tensor->is_variable();
    if (is_variable) {
      current->first_created = 0;
      current->last_used = operators_->size();
    } else {
      current->first_created = -1;
      current->last_used = -1;
    }
    current->needs_allocating = false;
    // Preallocated inputs have already been set up earlier, so skip them.
    const bool is_preallocated_input =
        (current->runtime_tensor->data.raw != nullptr);
    if (!is_preallocated_input) {
      TF_LITE_ENSURE_STATUS(InitializeRuntimeTensor(
          *current->flatbuffer_tensor, buffers, error_reporter_,
          current->runtime_tensor, nullptr));
    }
  }

  // First go through the inputs and figure out if they need to be allocated.
  for (size_t i = 0; i < subgraph_->inputs()->size(); ++i) {
    const int tensor_index = subgraph_->inputs()->Get(i);
    TensorInfo* current = &tensor_info[tensor_index];
    // Check for pre-allocated inputs.
    current->needs_allocating = (current->runtime_tensor->data.raw == nullptr);
    current->first_created = 0;
  }

  // Mark all outputs as persistent to the end of the invocation.
  for (size_t i = 0; i < subgraph_->outputs()->size(); ++i) {
    const int tensor_index = subgraph_->outputs()->Get(i);
    TensorInfo* current = &tensor_info[tensor_index];
    current->last_used = operators_->size() - 1;
  }

  // Figure out when the first and last use of each tensor is.
  for (int i = (operators_->size() - 1); i >= 0; --i) {
    const auto* op = operators_->Get(i);
    for (size_t n = 0; n < op->inputs()->size(); ++n) {
      const int tensor_index = op->inputs()->Get(n);
      TensorInfo* current = &tensor_info[tensor_index];
      if ((current->last_used == -1) || (current->last_used > i)) {
        current->last_used = i;
      }
    }
    for (size_t n = 0; n < op->outputs()->size(); ++n) {
      const int tensor_index = op->outputs()->Get(n);
      TensorInfo* current = &tensor_info[tensor_index];
      if ((current->first_created == -1) || (current->first_created < i)) {
        current->first_created = i;
      }
    }
  }

  // Work out which tensors need to be allocated.
  for (size_t i = 0; i < tensors_->size(); ++i) {
    TensorInfo* current = &tensor_info[i];
    const bool is_read_only =
        (current->first_created == -1) && (current->last_used != -1);
    const bool is_preallocated_input =
        (current->runtime_tensor->data.raw != nullptr);
    const bool has_partial_lifetime =
        !is_read_only &&
        ((current->first_created == -1) || (current->last_used == -1));
    if (has_partial_lifetime) {
      error_reporter_->Report(
          "Logic error in memory planner, tensor %d has an invalid lifetime",
          i);
      return kTfLiteError;
    }
    if (!is_read_only && !is_preallocated_input) {
      current->needs_allocating = true;
    }
  }

  uint8_t* aligned_arena = AlignPointerUp(arena_, kBufferAlignment);
  const size_t alignment_loss = (aligned_arena - arena_);

  int remaining_arena_size =
      arena_size_ - (memory_allocator_.GetDataSize() + alignment_loss);
  GreedyMemoryPlanner planner(aligned_arena, remaining_arena_size);

  // Add the tensors to our allocation plan.
  for (size_t i = 0; i < tensors_->size(); ++i) {
    TensorInfo* current = &tensor_info[i];
    if (current->needs_allocating) {
      size_t bytes_required;
      size_t type_size;
      TF_LITE_ENSURE_STATUS(BytesRequiredForTensor(*current->flatbuffer_tensor,
                                                   &bytes_required, &type_size,
                                                   error_reporter_));
      size_t aligned_bytes_required =
          AlignSizeUp(bytes_required, kBufferAlignment);
      planner.AddBuffer(error_reporter_, aligned_bytes_required,
                        current->first_created, current->last_used);
    }
  }

  // Make sure we have enough room.
  if (planner.GetMaximumMemorySize() > remaining_arena_size) {
    error_reporter_->Report(
        "Arena size is too small for activation buffers. Needed %d but only %d "
        "was available.",
        planner.GetMaximumMemorySize(), remaining_arena_size);
    return kTfLiteError;
  }

  // Figure out the actual memory addresses for each buffer, based on the plan.
  int planner_index = 0;
  for (size_t i = 0; i < tensors_->size(); ++i) {
    TensorInfo* current = &tensor_info[i];
    if (current->needs_allocating) {
      int offset;
      TF_LITE_ENSURE_STATUS(
          planner.GetOffsetForBuffer(error_reporter_, planner_index, &offset));
      current->runtime_tensor->data.uint8 = aligned_arena + offset;
      ++planner_index;
    }
    // Set default value for variable tensors:
    if (current->flatbuffer_tensor->is_variable()) {
      if (current->runtime_tensor->data.uint8 == nullptr) {
        error_reporter_->Report("Variable is not allocated");
        return kTfLiteError;
      }
      tflite::ResetVariableTensor(current->runtime_tensor);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::InitializeRuntimeTensor(
    const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result,
    uint8_t* preallocated_buffer) {
  // Make sure the serialized type is one we know how to deal with, and convert
  // it from a flatbuffer enum into a constant used by the kernel C API.
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &result->type, error_reporter));
  // Make sure we remember if the serialized tensor is designated as a variable.
  result->is_variable = flatbuffer_tensor.is_variable();

  // We need to figure out where the actual contents of this tensor are stored
  // in memory. We'll check to see if there's a serialized buffer (pretty much
  // the same as a constant op in TensorFlow) associated with this tensor first,
  // and if there is update the runtime structure to point to its location in
  // memory.
  result->data.raw = nullptr;
  result->bytes = 0;
  // First see if there's any buffer information in the serialized tensor.
  if (auto* buffer = (*buffers)[flatbuffer_tensor.buffer()]) {
    // If we've found a buffer, does it have any data?
    if (auto* array = buffer->data()) {
      // If it has any data, is the data size larger than zero?
      if (size_t array_size = array->size()) {
        // We've found a buffer with valid data, so update the runtime tensor
        // data structure to point to it.
        result->data.raw =
            const_cast<char*>(reinterpret_cast<const char*>(array->data()));
        // We set the data from a serialized buffer, so record tha.
        result->allocation_type = kTfLiteMmapRo;
      }
    }
    // TODO(petewarden): It's not clear in what circumstances we could have a
    // buffer in the serialized tensor, but it doesn't have any data in it. Is
    // that a validly-generated file, and if so what does it mean, or is it an
    // error condition? It would be good to tighten up the specification to make
    // it less ambiguous.
  }

  // TODO(petewarden): Some of these paths aren't getting enough testing
  // coverage, so we should figure out some tests that exercise them.
  if (!result->data.raw) {
    // The tensor contents haven't been set from a serialized buffer, so
    // make a note that they will be allocated from memory. The actual
    // allocation won't happen until later.
    result->allocation_type = kTfLiteArenaRw;
    if (preallocated_buffer != nullptr) {
      // If the client is supplying memory for the contents of the tensor
      // themselves, use it.
      // TODO(petewarden): Should we store the fact this is a client-allocated
      // buffer?
      result->data.raw = reinterpret_cast<char*>(preallocated_buffer);
    }
  }

  // Figure out what the size in bytes of the buffer is and store it.
  size_t type_size;
  TF_LITE_ENSURE_STATUS(BytesRequiredForTensor(
      flatbuffer_tensor, &result->bytes, &type_size, error_reporter));
  // Copy the shape of the tensor from the serialized data into the runtime
  // form. We have to allocate memory for this.
  result->dims =
      reinterpret_cast<TfLiteIntArray*>(memory_allocator_.AllocateFromTail(
          sizeof(int) * (flatbuffer_tensor.shape()->Length() + 1),
          sizeof(int)));
  result->dims->size = flatbuffer_tensor.shape()->Length();
  for (size_t n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
    result->dims->data[n] = flatbuffer_tensor.shape()->Get(n);
  }
  // Copy the quantization information from the serialized data.
  const auto* src_quantization = flatbuffer_tensor.quantization();
  if (src_quantization && src_quantization->scale() &&
      (src_quantization->scale()->size() > 0) &&
      src_quantization->zero_point() &&
      (src_quantization->zero_point()->size() > 0)) {
    result->params.scale = src_quantization->scale()->Get(0);
    // This magic handles issues with little-endianness.
    for (unsigned int b = 0; b < sizeof(int64_t); ++b)
      *(reinterpret_cast<char*>(&result->params.zero_point) + b) =
          *(reinterpret_cast<const char*>(
                src_quantization->zero_point()->Data()) +
            b);
    result->params.zero_point =
        flatbuffers::EndianScalar(result->params.zero_point);

    // Populate per-channel quantization params.
    int channels = src_quantization->scale()->size();
    TfLiteAffineQuantization* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            memory_allocator_.AllocateFromTail(sizeof(TfLiteAffineQuantization),
                                               sizeof(int)));
    int* zero_point_array =
        reinterpret_cast<int*>(memory_allocator_.AllocateFromTail(
            channels * sizeof(int) + sizeof(int), sizeof(int)));
    int* scale_array =
        reinterpret_cast<int*>(memory_allocator_.AllocateFromTail(
            channels * sizeof(float) + sizeof(int), sizeof(int)));
    zero_point_array[0] = channels;
    scale_array[0] = channels;
    int* zero_point_data = &zero_point_array[1];
    float* scale_data = reinterpret_cast<float*>(&scale_array[1]);
    for (int i = 0; i < channels; i++) {
      zero_point_data[i] = src_quantization->zero_point()->Get(i);
      scale_data[i] = src_quantization->scale()->Get(i);
    }
    quantization->scale = reinterpret_cast<TfLiteFloatArray*>(scale_array);
    quantization->zero_point =
        reinterpret_cast<TfLiteIntArray*>(zero_point_array);

    result->quantization = {kTfLiteAffineQuantization, quantization};
  }
  // Copy the name, if there is one.
  if (flatbuffer_tensor.name()->c_str() != nullptr) {
    result->name = flatbuffer_tensor.name()->c_str();
  } else {
    result->name = "<No name>";
  }
  // These aren't used by the micro flavor of TFL, so set them to defaults.
  result->allocation = nullptr;
  result->delegate = nullptr;
  result->buffer_handle = 0;
  result->data_is_stale = false;
  return kTfLiteOk;
}

}  // namespace tflite
