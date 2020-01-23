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
#include "tensorflow/lite/core/api/error_reporter.h"
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
struct AllocationInfo {
  size_t bytes;
  int first_created;
  int last_used;
  bool needs_allocating;
  void** output_ptr;
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

TfLiteStatus AllocateVariables(
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* flatbuffer_tensors,
    TfLiteTensor* runtime_tensors, SimpleMemoryAllocator* allocator) {
  for (size_t i = 0; i < flatbuffer_tensors->size(); ++i) {
    if (flatbuffer_tensors->Get(i)->is_variable()) {
      runtime_tensors[i].data.uint8 = allocator->AllocateFromTail(
          runtime_tensors[i].bytes, kBufferAlignment);
      // Allocation failure.
      if (runtime_tensors[i].data.uint8 == nullptr) {
        return kTfLiteError;
      }
    }
    tflite::ResetVariableTensor(&(runtime_tensors[i]));
  }
  return kTfLiteOk;
}

AllocationInfo* AllocateAndCalculateAllocationInfo(
    ErrorReporter* error_reporter, size_t allocation_info_size,
    const SubGraph* subgraph, TfLiteTensor* runtime_tensors,
    SimpleMemoryAllocator* allocator) {
  AllocationInfo* allocation_info = reinterpret_cast<AllocationInfo*>(
      allocator->AllocateFromTail(sizeof(AllocationInfo) * allocation_info_size,
                                  alignof(AllocationInfo)));
  if (allocation_info == nullptr) {
    error_reporter->Report(
        "Failed to allocate memory for allocation_info, %d bytes required",
        sizeof(TfLiteTensor) * allocation_info_size);
    return nullptr;
  }

  // Set up the runtime data structures for all tensors.
  for (size_t i = 0; i < allocation_info_size; ++i) {
    AllocationInfo* current = &allocation_info[i];
    // TfLiteTensor.uint8 field is deprecated so use .data field instead.
    current->output_ptr = &(runtime_tensors[i].data.data);
    current->bytes = runtime_tensors[i].bytes;
    current->first_created = -1;
    current->last_used = -1;
    current->needs_allocating = (runtime_tensors[i].data.raw == nullptr) &&
                                (!subgraph->tensors()->Get(i)->is_variable());
  }

  for (size_t i = 0; i < subgraph->inputs()->size(); ++i) {
    const int tensor_index = subgraph->inputs()->Get(i);
    AllocationInfo* current = &allocation_info[tensor_index];
    current->first_created = 0;
  }

  // Mark all outputs as persistent to the end of the invocation.
  for (size_t i = 0; i < subgraph->outputs()->size(); ++i) {
    const int tensor_index = subgraph->outputs()->Get(i);
    AllocationInfo* current = &allocation_info[tensor_index];
    current->last_used = subgraph->operators()->size() - 1;
  }

  // Figure out when the first and last use of each tensor is.
  for (int i = (subgraph->operators()->size() - 1); i >= 0; --i) {
    const auto* op = subgraph->operators()->Get(i);
    for (size_t n = 0; n < op->inputs()->size(); ++n) {
      const int tensor_index = op->inputs()->Get(n);
      AllocationInfo* current = &allocation_info[tensor_index];
      if (((current->last_used == -1) || (current->last_used > i))) {
        current->last_used = i;
      }
    }
    for (size_t n = 0; n < op->outputs()->size(); ++n) {
      const int tensor_index = op->outputs()->Get(n);
      AllocationInfo* current = &allocation_info[tensor_index];
      if ((current->first_created == -1) || (current->first_created < i)) {
        current->first_created = i;
      }
    }
  }

  // Work out which tensors need to be allocated.
  for (size_t i = 0; i < allocation_info_size; ++i) {
    AllocationInfo* current = &allocation_info[i];
    const bool is_read_only =
        (current->first_created == -1) && (current->last_used != -1);
    if (is_read_only) {
      current->needs_allocating = false;
    }
    const bool has_partial_lifetime =
        !is_read_only &&
        ((current->first_created == -1) || (current->last_used == -1));
    if (has_partial_lifetime && current->needs_allocating) {
      error_reporter->Report(
          "Logic error in memory planner, tensor %d has an invalid lifetime: "
          "first_created: %d, last_used: %d",
          i, current->first_created, current->last_used);
      return nullptr;
    }
  }  // namespace

  return allocation_info;
}  // namespace tflite

TfLiteStatus CreatePlan(ErrorReporter* error_reporter, MemoryPlanner* planner,
                        const AllocationInfo* allocation_info,
                        size_t allocation_info_size) {
  // Add the tensors to our allocation plan.
  for (size_t i = 0; i < allocation_info_size; ++i) {
    const AllocationInfo* current = &allocation_info[i];
    if (current->needs_allocating) {
      size_t aligned_bytes_required =
          AlignSizeUp(current->bytes, kBufferAlignment);
      TF_LITE_ENSURE_STATUS(
          planner->AddBuffer(error_reporter, aligned_bytes_required,
                             current->first_created, current->last_used));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CommitPlan(ErrorReporter* error_reporter, MemoryPlanner* planner,
                        uint8_t* starting_point,
                        AllocationInfo* allocation_info,
                        size_t allocation_info_size) {
  // Figure out the actual memory addresses for each buffer, based on the plan.
  int planner_index = 0;
  for (size_t i = 0; i < allocation_info_size; ++i) {
    AllocationInfo* current = &allocation_info[i];
    if (current->needs_allocating) {
      int offset = -1;
      TF_LITE_ENSURE_STATUS(
          planner->GetOffsetForBuffer(error_reporter, planner_index, &offset));
      *current->output_ptr = reinterpret_cast<void*>(starting_point + offset);
      ++planner_index;
    }
  }
  return kTfLiteOk;
}
}  // namespace

namespace internal {

TfLiteStatus InitializeRuntimeTensor(
    SimpleMemoryAllocator* allocator, const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result) {
  *result = {};
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

  // TFLM doesn't allow reshaping the tensor which requires dynamic memory
  // allocation so it is safe to drop the const qualifier. In the future, if we
  // really want to update the tensor shape, we can always pass in a new
  // TfLiteIntArray - especially we have to do so if the dimension is changed.
  result->dims = const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(flatbuffer_tensor.shape()));

  // Copy the quantization information from the serialized data.
  const auto* src_quantization = flatbuffer_tensor.quantization();
  if (src_quantization && src_quantization->scale() &&
      (src_quantization->scale()->size() > 0) &&
      src_quantization->zero_point() &&
      (src_quantization->zero_point()->size() > 0)) {
    // Always populate the TfLiteTensor.params field, even if there are
    // per-channel quantization parameters.
    result->params.scale = src_quantization->scale()->Get(0);
    // Note that the zero_point field in the FlatBuffers schema is a 64-bit
    // integer, but the zero_point field in the TfLiteQuantizationParams struct
    // is a 32-bit integer.
    result->params.zero_point =
        static_cast<int32_t>(src_quantization->zero_point()->Get(0));

    // Populate per-channel quantization params.
    int channels = src_quantization->scale()->size();
    TfLiteAffineQuantization* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            allocator->AllocateFromTail(sizeof(TfLiteAffineQuantization),
                                        alignof(TfLiteAffineQuantization)));
    quantization->zero_point =
        reinterpret_cast<TfLiteIntArray*>(allocator->AllocateFromTail(
            TfLiteIntArrayGetSizeInBytes(channels), alignof(TfLiteIntArray)));
    quantization->scale = reinterpret_cast<TfLiteFloatArray*>(
        allocator->AllocateFromTail(TfLiteFloatArrayGetSizeInBytes(channels),
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
  return kTfLiteOk;
}
}  // namespace internal

TfLiteStatus MicroAllocator::Init() {
  auto* subgraphs = model_->subgraphs();
  if (subgraphs->size() != 1) {
    error_reporter_->Report("Only 1 subgraph is currently supported.\n");
    return kTfLiteError;
  }
  subgraph_ = (*subgraphs)[0];
  tensors_ = subgraph_->tensors();
  operators_ = subgraph_->operators();

  context_->tensors_size = tensors_->size();
  context_->tensors =
      reinterpret_cast<TfLiteTensor*>(memory_allocator_->AllocateFromTail(
          sizeof(TfLiteTensor) * context_->tensors_size,
          alignof(TfLiteTensor)));
  if (context_->tensors == nullptr) {
    error_reporter_->Report(
        "Failed to allocate memory for context->tensors, %d bytes required",
        sizeof(TfLiteTensor) * context_->tensors_size);
  }

  // Initialize runtime tensors in context_ using the flatbuffer.
  for (size_t i = 0; i < tensors_->size(); ++i) {
    TfLiteStatus status = internal::InitializeRuntimeTensor(
        memory_allocator_, *tensors_->Get(i), model_->buffers(),
        error_reporter_, &context_->tensors[i]);
    if (status == kTfLiteError) {
      error_reporter_->Report("Failed to initialize tensor %d", i);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

MicroAllocator::MicroAllocator(TfLiteContext* context, const Model* model,
                               uint8_t* tensor_arena, size_t arena_size,
                               ErrorReporter* error_reporter)
    : model_(model), error_reporter_(error_reporter), context_(context) {
  uint8_t* aligned_arena = AlignPointerUp(tensor_arena, kBufferAlignment);
  size_t aligned_arena_size = tensor_arena + arena_size - aligned_arena;
  // Creates a root memory allocator managing the arena. The allocator itself
  // also locates in the arena buffer. This allocator doesn't need to be
  // destructed as it's the root allocator.
  SimpleMemoryAllocator* aligned_allocator =
      CreateInPlaceSimpleMemoryAllocator(aligned_arena, aligned_arena_size);
  memory_allocator_ = aligned_allocator;
  TfLiteStatus status = Init();
  // TODO(b/147871299): Consider improving this code. A better way of handling
  // failures in the constructor is to have a static function that returns a
  // pointer to the class. If allocation failed, a nullptr will be returned.
  if (status != kTfLiteOk) {
    error_reporter_->Report("MicroAllocator: Failed to initialize.");
    active_ = false;
  } else {
    active_ = true;
  }
}

TfLiteStatus MicroAllocator::AllocateNodeAndRegistrations(
    const OpResolver& op_resolver,
    NodeAndRegistration** node_and_registrations) {
  if (!active_) {
    return kTfLiteError;
  }

  auto* output = reinterpret_cast<NodeAndRegistration*>(
      memory_allocator_->AllocateFromTail(
          sizeof(NodeAndRegistration) * operators_->size(),
          alignof(NodeAndRegistration)));
  if (output == nullptr) {
    error_reporter_->Report(
        "Failed to allocate memory for node_and_registrations.");
    return kTfLiteError;
  }
  TfLiteStatus status = kTfLiteOk;
  auto* opcodes = model_->operator_codes();
  MicroBuiltinDataAllocator builtin_data_allocator(memory_allocator_);
  for (size_t i = 0; i < operators_->size(); ++i) {
    const auto* op = operators_->Get(i);
    size_t index = op->opcode_index();
    if (index >= opcodes->size()) {
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
    *node = {};
    node->inputs = inputs_array;
    node->outputs = outputs_array;
    // This is OK for now as temporary array is not in used.
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

  // Create static memory plan. AllocationInfo is needed for creating the plan
  // but is thrown away afterwards.
  {
    SimpleMemoryAllocator tmp_allocator =
        memory_allocator_->CreateChildAllocator();
    size_t allocation_info_size = tensors_->size();
    AllocationInfo* allocation_info = AllocateAndCalculateAllocationInfo(
        error_reporter_, allocation_info_size, subgraph_, context_->tensors,
        &tmp_allocator);
    if (allocation_info == nullptr) {
      return kTfLiteError;
    }

    uint8_t* aligned_arena = memory_allocator_->GetBuffer();
    size_t arena_size = memory_allocator_->GetMaxBufferSize();

    // Remaining arena size that memory planner can use for calculating offsets.
    // The remaining size should always be a positive number since the parent
    // allocator is always bigger than the child allocator.
    size_t remaining_arena_size = arena_size - tmp_allocator.GetDataSize();
    GreedyMemoryPlanner planner(aligned_arena, remaining_arena_size);
    TF_LITE_ENSURE_STATUS(CreatePlan(error_reporter_, &planner, allocation_info,
                                     allocation_info_size));

    // Actual size available for placing tensors. This includes memory held by
    // the tensor info array, which will be released.
    size_t actual_available_arena_size =
        arena_size - memory_allocator_->GetDataSize();
    // Make sure we have enough room.
    // TODO(b/147871342): make GetMaximumMemorySize return size_t.
    // int is more than enough to hold arena_size since we're only dealing with
    // at most several megabytes memory.
    if (planner.GetMaximumMemorySize() >
        static_cast<int>(actual_available_arena_size)) {
      error_reporter_->Report(
          "Arena size is too small for activation buffers. Needed %d but only "
          "%d was available.",
          planner.GetMaximumMemorySize(), remaining_arena_size);
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(CommitPlan(error_reporter_, &planner, aligned_arena,
                                     allocation_info, allocation_info_size));
  }

  // Data in variables need to be kept for the next invocation so allocating
  // them from the tail (persistent area).
  if (AllocateVariables(tensors_, context_->tensors, memory_allocator_) !=
      kTfLiteOk) {
    error_reporter_->Report(
        "Failed to allocate variables. Please increase arena size.");
    return kTfLiteError;
  }

  active_ = false;
  return kTfLiteOk;
}

}  // namespace tflite
