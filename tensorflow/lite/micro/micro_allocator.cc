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

#include "tensorflow/lite/micro/micro_allocator.h"

#include <cstddef>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"

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

// If building with GNU clib from GCC 4.8.x or lower, `max_align_t` is not a
// member of `std`. If using a newer version of clib, we import `max_align_t`
// into the local anonymous namespace to be able to use it like the global
// `max_align_t` from the older clib.
#if defined(__GNUC__) && defined(__GNUC_PREREQ)
#if __GNUC_PREREQ(4, 9)
using std::max_align_t;
#endif
#else
// We assume other compiler/clib configurations don't have this issue.
using std::max_align_t;
#endif

class MicroBuiltinDataAllocator : public BuiltinDataAllocator {
 public:
  explicit MicroBuiltinDataAllocator(SimpleMemoryAllocator* memory_allocator)
      : memory_allocator_(memory_allocator) {}

  void* Allocate(size_t size) override {
    // Align to an address that is proper for all primitive types, but no more
    // than the size.
    return memory_allocator_->AllocateFromTail(
        size, std::min(size, alignof(max_align_t)));
  }
  void Deallocate(void* data) override {
    // Do not deallocate, builtin data needs to be available for the life time
    // of the model.
  }

 private:
  SimpleMemoryAllocator* memory_allocator_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

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
          sizeof(TfLiteTensor) * context_->tensors_size,
          alignof(TfLiteTensor)));
  if (context_->tensors == nullptr) {
    error_reporter_->Report(
        "Failed to allocate memory for context->tensors, %d bytes required",
        sizeof(TfLiteTensor) * context_->tensors_size);
  }
  active_ = true;
}

TfLiteStatus MicroAllocator::AllocateNodeAndRegistrations(
    const OpResolver& op_resolver,
    NodeAndRegistration** node_and_registrations) {
  if (!active_) {
    return kTfLiteError;
  }

  auto* output =
      reinterpret_cast<NodeAndRegistration*>(memory_allocator_.AllocateFromTail(
          sizeof(NodeAndRegistration) * operators_->size(),
          alignof(NodeAndRegistration)));
  if (output == nullptr) {
    error_reporter_->Report(
        "Failed to allocate memory for node_and_registrations.");
    return kTfLiteError;
  }
  TfLiteStatus status = kTfLiteOk;
  auto* opcodes = model_->operator_codes();
  MicroBuiltinDataAllocator builtin_data_allocator(&memory_allocator_);
  for (size_t i = 0; i < operators_->size(); ++i) {
    const auto* op = operators_->Get(i);
    size_t index = op->opcode_index();
    if (index < 0 || index >= opcodes->size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      return kTfLiteError;
    }
    auto* opcode = (*opcodes)[index];
    status = GetRegistrationFromOpCode(opcode, op_resolver, error_reporter_,
                                       &(output[i].registration));
    if (status != kTfLiteOk) {
      error_reporter_->Report("Failed to get registration from op code % d\n ",
                              opcode);
      return status;
    }
    const auto* registration = output[i].registration;
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      return kTfLiteError;
    }
    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Unsupported behavior: found builtin operator %s with custom "
          "options.\n",
          EnumNameBuiltinOperator(op_type));
      return kTfLiteError;
    }

    const char* custom_data = nullptr;
    size_t custom_data_size = 0;
    unsigned char* builtin_data = nullptr;
    if (op->custom_options()) {
      custom_data = reinterpret_cast<const char*>(op->custom_options()->data());
      custom_data_size = op->custom_options()->size();
    } else {
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &builtin_data_allocator,
                                        (void**)(&builtin_data)));
    }

    // Disregard const qualifier to workaround with existing API.
    TfLiteIntArray* inputs_array = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(op->inputs()));
    TfLiteIntArray* outputs_array = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(op->outputs()));

    TfLiteNode* node = &(output[i].node);
    node->inputs = inputs_array;
    node->outputs = outputs_array;
    // This is OK for now as temporary array is not in used.
    // TODO(wangtz): Support scratch buffers.
    node->temporaries = nullptr;
    node->user_data = nullptr;  // Will be filled in after `init`
    node->builtin_data = reinterpret_cast<void*>(builtin_data);
    node->custom_initial_data = custom_data;
    node->custom_initial_data_size = custom_data_size;
    node->delegate = nullptr;
  }
  *node_and_registrations = output;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::FinishTensorAllocation() {
  if (!active_) {
    return kTfLiteError;
  }

  // Initialize runtime tensors in context_ using the flatbuffer.
  for (size_t i = 0; i < tensors_->size(); ++i) {
    TF_LITE_ENSURE_STATUS(
        InitializeRuntimeTensor(*tensors_->Get(i), model_->buffers(),
                                error_reporter_, &context_->tensors[i]));
  }

  // tensor_info is only used in this function.
  SimpleMemoryAllocator tmp_allocator =
      memory_allocator_.CreateChildAllocator();
  TensorInfo* tensor_info =
      reinterpret_cast<TensorInfo*>(tmp_allocator.AllocateFromTail(
          sizeof(TensorInfo) * tensors_->size(), alignof(TensorInfo)));
  if (tensor_info == nullptr) {
    error_reporter_->Report(
        "Failed to allocate memory for tensor_info, %d bytes required",
        sizeof(TfLiteTensor) * context_->tensors_size);
    return kTfLiteError;
  }

  // Set up the runtime data structures for all tensors.
  for (size_t i = 0; i < tensors_->size(); ++i) {
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

  // Remaining arena size that memory planner can use for calculating offsets.
  int remaining_arena_size =
      arena_size_ - (tmp_allocator.GetDataSize() + alignment_loss);
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
      TF_LITE_ENSURE_STATUS(
          planner.AddBuffer(error_reporter_, aligned_bytes_required,
                            current->first_created, current->last_used));
    }
  }

  // Actual size available for placing tensors. This includes memory held by the
  // tensor info array, which will be released.
  int actual_available_arena_size =
      arena_size_ - (memory_allocator_.GetDataSize() + alignment_loss);
  // Make sure we have enough room.
  if (planner.GetMaximumMemorySize() > actual_available_arena_size) {
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
  }

  // Copy default value for variable tensors. Note that this will overwrite
  // the arena planner data so GetOffsetForBuffer will return wrong
  // result.
  for (size_t i = 0; i < tensors_->size(); ++i) {
    TensorInfo* current = &tensor_info[i];
    // Set default value for variable tensors:
    if (current->flatbuffer_tensor->is_variable()) {
      if (current->runtime_tensor->data.uint8 == nullptr) {
        error_reporter_->Report("Variable is not allocated");
        return kTfLiteError;
      }
      tflite::ResetVariableTensor(current->runtime_tensor);
    }
  }

  active_ = false;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::InitializeRuntimeTensor(
    const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result) {
  if (!active_) {
    return kTfLiteError;
  }

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
      if (array->size()) {
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
  }

  // Figure out what the size in bytes of the buffer is and store it.
  size_t type_size;
  TF_LITE_ENSURE_STATUS(BytesRequiredForTensor(
      flatbuffer_tensor, &result->bytes, &type_size, error_reporter));
  // Copy the shape of the tensor from the serialized data into the runtime
  // form. We have to allocate memory for this.
  result->dims =
      reinterpret_cast<TfLiteIntArray*>(memory_allocator_.AllocateFromTail(
          TfLiteIntArrayGetSizeInBytes(flatbuffer_tensor.shape()->Length()),
          alignof(TfLiteIntArray)));
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
            memory_allocator_.AllocateFromTail(
                sizeof(TfLiteAffineQuantization),
                alignof(TfLiteAffineQuantization)));
    quantization->zero_point =
        reinterpret_cast<TfLiteIntArray*>(memory_allocator_.AllocateFromTail(
            TfLiteIntArrayGetSizeInBytes(channels), alignof(TfLiteIntArray)));
    quantization->scale =
        reinterpret_cast<TfLiteFloatArray*>(memory_allocator_.AllocateFromTail(
            TfLiteFloatArrayGetSizeInBytes(channels),
            alignof(TfLiteFloatArray)));
    quantization->zero_point->size = channels;
    quantization->scale->size = channels;
    int* zero_point_data = quantization->zero_point->data;
    float* scale_data = quantization->scale->data;
    for (int i = 0; i < channels; i++) {
      zero_point_data[i] = src_quantization->zero_point()->Get(i);
      scale_data[i] = src_quantization->scale()->Get(i);
    }
    // TODO(rocky): Need to add a micro_allocator test case that fails when
    // this is not copied:
    quantization->quantized_dimension = src_quantization->quantized_dimension();

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
