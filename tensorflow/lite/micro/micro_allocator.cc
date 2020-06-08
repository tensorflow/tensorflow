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
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/memory_planner.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {
// Used to hold information used during allocation calculations.
struct AllocationInfo {
  size_t bytes;
  void** output_ptr;
  int first_created;
  int last_used;
  int32_t offline_offset;
  bool needs_allocating;
};

// We align tensor buffers to 16-byte boundaries, since this is a common
// requirement for SIMD extensions.
constexpr int kBufferAlignment = 16;

constexpr char kOfflineMemAllocMetadata[] = "OfflineMemoryAllocation";

// Instance of a zero-length int to pass as tensor dims for a flatbuffer
// Tensor with no shape. Note that the second member of a TfLiteArray is a
// flexible array member, which is not strictly valid C++. However it is
// supported by both GCC and clang, as long as the flexible array element is not
// initialized, which is ok in this case as it should never be accessed.
// Declaring this as constexpr causes build errors with clang, as it requires
// the flexible array element to be initialized.
const TfLiteIntArray kZeroLengthIntArray = {0};

class MicroBuiltinDataAllocator : public BuiltinDataAllocator {
 public:
  explicit MicroBuiltinDataAllocator(SimpleMemoryAllocator* memory_allocator)
      : memory_allocator_(memory_allocator) {}

  void* Allocate(size_t size, size_t alignment_hint) override {
    return memory_allocator_->AllocateFromTail(size, alignment_hint);
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
      runtime_tensors[i].data.data = allocator->AllocateFromTail(
          runtime_tensors[i].bytes, kBufferAlignment);
      // Allocation failure.
      if (runtime_tensors[i].data.data == nullptr) {
        return kTfLiteError;
      }
    }
    tflite::ResetVariableTensor(&(runtime_tensors[i]));
  }
  return kTfLiteOk;
}

#if !defined(__clang__)
// Helper function to check flatbuffer metadata correctness. This function is
// not called by default. Hence it's not linked in to the final binary code.
TfLiteStatus CheckOfflinePlannedOffsets(const Model* model,
                                        ErrorReporter* error_reporter) {
  // Suppress compile warning for unused function
  (void)CheckOfflinePlannedOffsets;

  if (model->metadata()) {
    for (size_t i = 0; i < model->metadata()->size(); ++i) {
      auto metadata = model->metadata()->Get(i);
      if (strncmp(metadata->name()->c_str(), kOfflineMemAllocMetadata,
                  strlen(kOfflineMemAllocMetadata)) == 0) {
        auto* subgraphs = model->subgraphs();
        const SubGraph* subgraph = (*subgraphs)[0];
        const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors =
            subgraph->tensors();
        const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
            model->buffers();
        int nbr_tflite_tensors = tensors->size();
        auto* buffer = (*buffers)[metadata->buffer()];
        auto* array = buffer->data();
        const uint32_t* metadata_buffer = (uint32_t*)array->data();
        int version = metadata_buffer[0];
        int subgraph_idx = metadata_buffer[1];
        const int nbr_offline_offsets = metadata_buffer[2];
        int* offline_planner_offsets = (int*)&metadata_buffer[3];

        TF_LITE_REPORT_ERROR(error_reporter, "==== Model metadata info: =====");
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Offline planner metadata found, version %d, "
                             "subgraph %d, nbr offline offsets %d",
                             version, subgraph_idx, nbr_offline_offsets);
        for (int j = 0; j < nbr_offline_offsets; ++j) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Offline planner tensor index %d, offline offset: %d", j,
              offline_planner_offsets[j]);
        }

        if (version != 1) {
          TF_LITE_REPORT_ERROR(error_reporter, "Version not supported! (%d)\n",
                               version);
          return kTfLiteError;
        }
        if (subgraph_idx != 0) {
          TF_LITE_REPORT_ERROR(error_reporter,
                               "Only 1 subgraph supported! Subgraph idx (%d)\n",
                               subgraph_idx);
          return kTfLiteError;
        }
        if (nbr_tflite_tensors != nbr_offline_offsets) {
          TF_LITE_REPORT_ERROR(error_reporter,
                               "Nbr of offline buffer offsets (%d) in metadata "
                               "not equal nbr tensors (%d)\n",
                               nbr_offline_offsets, nbr_tflite_tensors);
          return kTfLiteError;
        }
      }
    }
  }
  return kTfLiteOk;
}
#endif

// A helper class to construct AllocationInfo array. This array contains the
// lifetime of tensors / scratch_buffer and will be used to calculate the memory
// plan. Methods need to be called in order from `Init`, `Add*`, to `Finish`.
class AllocationInfoBuilder {
 public:
  AllocationInfoBuilder(ErrorReporter* reporter,
                        SimpleMemoryAllocator* allocator)
      : reporter_(reporter), allocator_(allocator) {}

  // Initializes the builder by allocating AllocationInfo array from the
  // simple memory allocator.
  TfLiteStatus Init(size_t tensor_count, size_t scratch_buffer_count) {
    tensor_count_ = tensor_count;
    buffer_count_ = scratch_buffer_count;
    return Allocate();
  }

  // Check if model contains offline planned buffer offsets.
  //  - If there's no metadata available, offline_planner_offsets is not set
  //  - If there's metadata available, offline_planner_offsets will point to the
  //    first offset in the metadata buffer list.
  TfLiteStatus GetOfflinePlannedOffsets(const Model* model,
                                        int32_t** offline_planner_offsets);

  // Add allocaiton information for the tensors.
  TfLiteStatus AddTensors(const SubGraph* subgraph, int32_t* offline_offsets,
                          TfLiteTensor* runtime_tensors);

  // Add allocation information for the scratch buffers.
  TfLiteStatus AddScratchBuffers(internal::ScratchBufferHandle* buffer_handles);

  // Returns a pointer to the built AllocationInfo array.
  const AllocationInfo* Finish() const { return info_; }
  size_t Size() const { return tensor_count_ + buffer_count_; }

 private:
  // Allocate the output AllocationInfo array from the allocator_;
  TfLiteStatus Allocate();

  ErrorReporter* reporter_ = nullptr;
  SimpleMemoryAllocator* allocator_ = nullptr;
  size_t tensor_count_ = 0;
  size_t buffer_count_ = 0;
  AllocationInfo* info_ = nullptr;
};

TfLiteStatus AllocationInfoBuilder::Allocate() {
  size_t bytes = sizeof(AllocationInfo) * Size();
  info_ = reinterpret_cast<AllocationInfo*>(
      allocator_->AllocateFromTail(bytes, alignof(AllocationInfo)));
  if (info_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        reporter_,
        "Failed to allocate memory for allocation_info, %d bytes required",
        bytes);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus AllocationInfoBuilder::AddTensors(const SubGraph* subgraph,
                                               int32_t* offline_offsets,
                                               TfLiteTensor* runtime_tensors) {
  // Set up allocation info for all tensors.
  for (size_t i = 0; i < tensor_count_; ++i) {
    AllocationInfo* current = &info_[i];
    // TfLiteTensor.uint8 field is deprecated so use .data field instead.
    current->output_ptr = &(runtime_tensors[i].data.data);
    current->bytes = runtime_tensors[i].bytes;
    current->first_created = -1;
    current->last_used = -1;
    current->needs_allocating = (runtime_tensors[i].data.data == nullptr) &&
                                (!subgraph->tensors()->Get(i)->is_variable());
    if (offline_offsets) {
      current->offline_offset = offline_offsets[i];
    } else {
      current->offline_offset = kOnlinePlannedBuffer;
    }
  }

  for (size_t i = 0; i < subgraph->inputs()->size(); ++i) {
    const int tensor_index = subgraph->inputs()->Get(i);
    AllocationInfo* current = &info_[tensor_index];
    current->first_created = 0;
  }

  // Mark all outputs as persistent to the end of the invocation.
  for (size_t i = 0; i < subgraph->outputs()->size(); ++i) {
    const int tensor_index = subgraph->outputs()->Get(i);
    AllocationInfo* current = &info_[tensor_index];
    current->last_used = subgraph->operators()->size() - 1;
  }

  // Figure out when the first and last use of each tensor is.
  for (int i = (subgraph->operators()->size() - 1); i >= 0; --i) {
    const auto* op = subgraph->operators()->Get(i);
    for (size_t n = 0; n < op->inputs()->size(); ++n) {
      const int tensor_index = op->inputs()->Get(n);
      AllocationInfo* current = &info_[tensor_index];
      if (((current->last_used == -1) || (current->last_used < i))) {
        current->last_used = i;
      }
    }
    for (size_t n = 0; n < op->outputs()->size(); ++n) {
      const int tensor_index = op->outputs()->Get(n);
      AllocationInfo* current = &info_[tensor_index];
      if ((current->first_created == -1) || (current->first_created > i)) {
        current->first_created = i;
      }
    }
  }

  // Work out which tensors need to be allocated.
  for (size_t i = 0; i < tensor_count_; ++i) {
    AllocationInfo* current = &info_[i];
    const bool is_read_only =
        (current->first_created == -1) && (current->last_used != -1);
    if (is_read_only) {
      current->needs_allocating = false;
    }
    const bool has_partial_lifetime =
        !is_read_only &&
        ((current->first_created == -1) || (current->last_used == -1));
    if (has_partial_lifetime && current->needs_allocating) {
      TF_LITE_REPORT_ERROR(
          reporter_,
          "Logic error in memory planner, tensor %d has an invalid lifetime: "
          "first_created: %d, last_used: %d",
          i, current->first_created, current->last_used);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

// The tensor offsets will be encoded in the metadata:[Metadata] field of the
// Model. The following encoding applies:
//
// | Metadata component |                 Value                                |
// |    name:string     | “OfflineMemoryAllocation”                            |
// |    buffer:unit     | Index of buffer containing memory allocation data    |
//
// The buffer contents for the memory allocation is a list of 32-bit integers.
// The number of tensors, n, must be equal to the number of tensors defined in
// the model. The following encoding applies:
//
// |  Offset |                            Value                                |
// |    0    | Offline allocation format version – set to 0                    |
// |    1    | Subgraph index to which this allocation applies                 |
// |    2    | Number offsets following: n                                     |
// |    3    | Arena byte offset of tensor #0 or -1 to allocate at runtime     |
// |    4    | Arena byte offset of tensor #1 or -1 to allocate at runtime     |
// | 3+(n-1) | Arena byte offset of tensor #(n-1) or -1 to allocate at runtime |
TfLiteStatus AllocationInfoBuilder::GetOfflinePlannedOffsets(
    const Model* model, int32_t** offline_planner_offsets) {
  if (model->metadata()) {
    for (size_t i = 0; i < model->metadata()->size(); ++i) {
      auto metadata = model->metadata()->Get(i);
      if (strncmp(metadata->name()->c_str(), kOfflineMemAllocMetadata,
                  strlen(kOfflineMemAllocMetadata)) == 0) {
        const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
            model->buffers();
        auto* buffer = (*buffers)[metadata->buffer()];
        auto* array = buffer->data();
        const uint32_t* metadata_buffer = (uint32_t*)array->data();
        const size_t nbr_tensors = (size_t)metadata_buffer[2];
        *offline_planner_offsets = (int32_t*)&metadata_buffer[3];

        if (tensor_count_ != nbr_tensors) {
          TF_LITE_REPORT_ERROR(reporter_,
                               "Nbr of offline buffer offsets (%d) in metadata "
                               "not equal nbr tensors (%d)\n",
                               nbr_tensors, tensor_count_);
          return kTfLiteError;
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus AllocationInfoBuilder::AddScratchBuffers(
    internal::ScratchBufferHandle* buffer_handles) {
  // Set up allocation info for buffers.
  for (size_t i = tensor_count_; i < tensor_count_ + buffer_count_; ++i) {
    AllocationInfo* current = &info_[i];
    internal::ScratchBufferHandle* handle =
        &(buffer_handles[i - tensor_count_]);
    current->output_ptr = reinterpret_cast<void**>(&handle->data);
    current->bytes = handle->bytes;
    current->first_created = handle->node_idx;
    current->last_used = handle->node_idx;
    current->needs_allocating = true;
    current->offline_offset = kOnlinePlannedBuffer;
  }
  return kTfLiteOk;
}

TfLiteStatus CreatePlan(ErrorReporter* error_reporter,
                        GreedyMemoryPlanner* planner,
                        const AllocationInfo* allocation_info,
                        size_t allocation_info_size) {
  // Add the tensors to our allocation plan.
  for (size_t i = 0; i < allocation_info_size; ++i) {
    const AllocationInfo* current = &allocation_info[i];
    if (current->needs_allocating) {
      size_t aligned_bytes_required =
          AlignSizeUp(current->bytes, kBufferAlignment);
      if (current->offline_offset == kOnlinePlannedBuffer) {
        TF_LITE_ENSURE_STATUS(
            planner->AddBuffer(error_reporter, aligned_bytes_required,
                               current->first_created, current->last_used));
      } else {
        TF_LITE_ENSURE_STATUS(
            planner->AddBuffer(error_reporter, aligned_bytes_required,
                               current->first_created, current->last_used,
                               current->offline_offset));
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CommitPlan(ErrorReporter* error_reporter, MemoryPlanner* planner,
                        uint8_t* starting_point,
                        const AllocationInfo* allocation_info,
                        size_t allocation_info_size) {
  // Figure out the actual memory addresses for each buffer, based on the plan.
  int planner_index = 0;
  for (size_t i = 0; i < allocation_info_size; ++i) {
    const AllocationInfo* current = &allocation_info[i];
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

TfLiteStatus InitializeTfLiteTensorFromFlatbuffer(
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
        result->data.data =
            const_cast<void*>(static_cast<const void*>(array->data()));
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
  if (result->data.data == nullptr) {
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
  // allocation so it is safe to drop the const qualifier. In the future, if
  // we really want to update the tensor shape, we can always pass in a new
  // TfLiteIntArray - especially we have to do so if the dimension is changed.
  if (flatbuffer_tensor.shape() == nullptr) {
    // flatbuffer_tensor.shape() can return a nullptr in the case of a scalar
    // tensor.
    result->dims = const_cast<TfLiteIntArray*>(&kZeroLengthIntArray);
  } else {
    result->dims = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(flatbuffer_tensor.shape()));
  }

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
    if (quantization == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Unable to allocate TfLiteAffineQuantization.\n");
      return kTfLiteError;
    }
    quantization->zero_point =
        reinterpret_cast<TfLiteIntArray*>(allocator->AllocateFromTail(
            TfLiteIntArrayGetSizeInBytes(channels), alignof(TfLiteIntArray)));
    if (quantization->zero_point == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Unable to allocate quantization->zero_point.\n");
      return kTfLiteError;
    }

    quantization->scale = reinterpret_cast<TfLiteFloatArray*>(
        allocator->AllocateFromTail(TfLiteFloatArrayGetSizeInBytes(channels),
                                    alignof(TfLiteFloatArray)));
    if (quantization->scale == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Unable to allocate quantization->scale.\n");
      return kTfLiteError;
    }

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
  if (flatbuffer_tensor.name() != nullptr) {
    result->name = flatbuffer_tensor.name()->c_str();
  }
  return kTfLiteOk;
}

}  // namespace internal

MicroAllocator::MicroAllocator(TfLiteContext* context, const Model* model,
                               uint8_t* tensor_arena, size_t arena_size,
                               ErrorReporter* error_reporter)
    : model_(model),
      context_(context),
      error_reporter_(error_reporter),
      active_(false) {
  uint8_t* aligned_arena = AlignPointerUp(tensor_arena, kBufferAlignment);
  if (aligned_arena != tensor_arena) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "%d bytes lost due to alignment. To avoid this loss, please make sure "
        "the tensor_arena is 16 bytes aligned.",
        aligned_arena - tensor_arena);
  }
  size_t aligned_arena_size = tensor_arena + arena_size - aligned_arena;
  // Creates a root memory allocator managing the arena. The allocator itself
  // also locates in the arena buffer. This allocator doesn't need to be
  // destructed as it's the root allocator.
  memory_allocator_ = SimpleMemoryAllocator::Create(
      error_reporter, aligned_arena, aligned_arena_size);
}

MicroAllocator::MicroAllocator(TfLiteContext* context, const Model* model,
                               SimpleMemoryAllocator* memory_allocator,
                               ErrorReporter* error_reporter)
    : memory_allocator_(memory_allocator),
      model_(model),
      context_(context),
      error_reporter_(error_reporter),
      active_(false) {}

MicroAllocator::~MicroAllocator() {}

TfLiteStatus MicroAllocator::Init() {
  TfLiteStatus status = InitGraphAndContextTensorData();
  // TODO(b/147871299): Consider improving this code. A better way of handling
  // failures in the constructor is to have a static function that returns a
  // pointer to the class. If allocation failed, a nullptr will be returned.
  if (status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "MicroAllocator: Failed to initialize.");
    active_ = false;
  } else {
    active_ = true;
  }
  return status;
}

TfLiteStatus MicroAllocator::PrepareFromFlatbuffer(
    const MicroOpResolver& op_resolver,
    NodeAndRegistration** node_and_registrations) {
  if (!active_) {
    return kTfLiteError;
  }
  TF_LITE_ENSURE_STATUS(AllocateNodeAndRegistrations(node_and_registrations));
  TF_LITE_ENSURE_STATUS(PrepareNodeAndRegistrationDataFromFlatbuffer(
      op_resolver, *node_and_registrations));
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::FinishTensorAllocation() {
  if (!active_) {
    return kTfLiteError;
  }

  // Create static memory plan
  // 1. Calculate AllocationInfo to know the lifetime of each tensor/buffer.
  // 2. Add them into the planner (such as the GreedyMemoryPlanner).
  // 3. Static memory planning using the planner.
  // 4. Set tensor/buffer pointers based on the offsets from the previous step.
  // Note that AllocationInfo is only needed for creating the plan. It will be
  // thrown away when the child allocator (tmp_allocator) goes out of scope.
  {
    SimpleMemoryAllocator tmp_allocator(error_reporter_,
                                        memory_allocator_->GetHead(),
                                        memory_allocator_->GetTail());

    AllocationInfoBuilder builder(error_reporter_, &tmp_allocator);
    TF_LITE_ENSURE_STATUS(
        builder.Init(subgraph_->tensors()->size(), scratch_buffer_count_));
    int32_t* offline_planner_offsets = nullptr;
    TF_LITE_ENSURE_STATUS(
        builder.GetOfflinePlannedOffsets(model_, &offline_planner_offsets));
    TF_LITE_ENSURE_STATUS(builder.AddTensors(subgraph_, offline_planner_offsets,
                                             context_->tensors));
    TF_LITE_ENSURE_STATUS(builder.AddScratchBuffers(scratch_buffer_handles_));
    const AllocationInfo* allocation_info = builder.Finish();

    // Remaining arena size that memory planner can use for calculating offsets.
    size_t remaining_arena_size = tmp_allocator.GetAvailableMemory();
    uint8_t* planner_arena =
        tmp_allocator.AllocateFromHead(remaining_arena_size, /*alignment=*/1);
    TF_LITE_ENSURE(error_reporter_, planner_arena != nullptr);
    GreedyMemoryPlanner planner(planner_arena, remaining_arena_size);
    TF_LITE_ENSURE_STATUS(
        CreatePlan(error_reporter_, &planner, allocation_info, builder.Size()));

    size_t actual_available_arena_size =
        memory_allocator_->GetAvailableMemory();
    // Make sure we have enough arena size.
    if (planner.GetMaximumMemorySize() > actual_available_arena_size) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Arena size is too small for activation buffers. Needed %d but only "
          "%d was available.",
          planner.GetMaximumMemorySize(), actual_available_arena_size);
      return kTfLiteError;
    }

    // Commit the plan.
    TF_LITE_ENSURE_STATUS(CommitPlan(error_reporter_, &planner,
                                     memory_allocator_->GetHead(),
                                     allocation_info, builder.Size()));
    // Allocate the planned area, so the allocator knows it's used.
    uint8_t* allocated_tensor_memory =
        memory_allocator_->AllocateFromHead(planner.GetMaximumMemorySize(),
                                            /*alignment=*/1);
    TF_LITE_ENSURE(error_reporter_, allocated_tensor_memory != nullptr);
  }

  // Data in variables need to be kept for the next invocation so allocating
  // them from the tail (persistent area).
  if (AllocateVariables(subgraph_->tensors(), context_->tensors,
                        memory_allocator_) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate variables. Please increase arena size.");
    return kTfLiteError;
  }

  active_ = false;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::AllocatePersistentBuffer(size_t bytes,
                                                      void** ptr) {
  uint8_t* data = memory_allocator_->AllocateFromTail(bytes, kBufferAlignment);
  if (data == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed to allocate persistent buffer of size %d",
                         bytes);
    return kTfLiteError;
  }
  (*ptr) = data;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::RequestScratchBufferInArena(int node_id,
                                                         size_t bytes,
                                                         int* buffer_idx) {
  // A sanity check to make sure scratch_buffer_handles_ is contiguous i.e.
  // scratch_buffer_handles_ is pointing to the last allocation from memory
  // allocator.
  if (scratch_buffer_handles_ != nullptr &&
      reinterpret_cast<uint8_t*>(scratch_buffer_handles_) !=
          memory_allocator_->GetTail()) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Internal error: AllocateFromTail can not be called "
                         "between two RequestScratchBufferInArena calls.");
    return kTfLiteError;
  }

  internal::ScratchBufferHandle* handle =
      reinterpret_cast<internal::ScratchBufferHandle*>(
          memory_allocator_->AllocateFromTail(
              sizeof(internal::ScratchBufferHandle),
              alignof(internal::ScratchBufferHandle)));
  if (handle == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed to register scratch buffer handle for node %s",
                         node_id);
    return kTfLiteError;
  }
  *handle = {};
  handle->bytes = bytes;
  handle->node_idx = node_id;
  *buffer_idx = scratch_buffer_count_;
  scratch_buffer_count_ += 1;
  // scratch_buffer_handles_ is in reverse order. The following code ensures
  // that scratch_buffers[0] is pointing to the newly allocated handle.
  scratch_buffer_handles_ = handle;
  return kTfLiteOk;
}

void* MicroAllocator::GetScratchBuffer(int buffer_idx) const {
  if (static_cast<size_t>(buffer_idx) >= scratch_buffer_count_) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Buffer %d not found. %d buffers available.",
                         buffer_idx, scratch_buffer_count_);
    return nullptr;
  }
  // scratch_buffer_handles_ is in reverse order.
  return scratch_buffer_handles_[scratch_buffer_count_ - buffer_idx - 1].data;
}

size_t MicroAllocator::used_bytes() const {
  if (active_) {
    return 0;
  }
  return memory_allocator_->GetUsedBytes();
}

TfLiteStatus MicroAllocator::InitGraphAndContextTensorData() {
  auto* subgraphs = model_->subgraphs();
  if (subgraphs->size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Only 1 subgraph is currently supported.\n");
    return kTfLiteError;
  }
  subgraph_ = (*subgraphs)[0];

  TF_LITE_ENSURE_STATUS(AllocateTfLiteTensorArray());
  TF_LITE_ENSURE_STATUS(PopulateTfLiteTensorArrayFromFlatbuffer());

  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::AllocateTfLiteTensorArray() {
  context_->tensors_size = subgraph_->tensors()->size();
  context_->tensors =
      reinterpret_cast<TfLiteTensor*>(memory_allocator_->AllocateFromTail(
          sizeof(TfLiteTensor) * context_->tensors_size,
          alignof(TfLiteTensor)));
  if (context_->tensors == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory for context->tensors, %d bytes required",
        sizeof(TfLiteTensor) * context_->tensors_size);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::PopulateTfLiteTensorArrayFromFlatbuffer() {
  // Initialize tensors in context_ using the flatbuffer for quantization data.
  for (size_t i = 0; i < subgraph_->tensors()->size(); ++i) {
    TfLiteStatus status = internal::InitializeTfLiteTensorFromFlatbuffer(
        memory_allocator_, *subgraph_->tensors()->Get(i), model_->buffers(),
        error_reporter_, &context_->tensors[i]);
    if (status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter_, "Failed to initialize tensor %d",
                           i);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::AllocateNodeAndRegistrations(
    NodeAndRegistration** node_and_registrations) {
  NodeAndRegistration* output = reinterpret_cast<NodeAndRegistration*>(
      memory_allocator_->AllocateFromTail(
          sizeof(NodeAndRegistration) * subgraph_->operators()->size(),
          alignof(NodeAndRegistration)));
  if (output == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory for node_and_registrations.");
    return kTfLiteError;
  }
  *node_and_registrations = output;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::PrepareNodeAndRegistrationDataFromFlatbuffer(
    const MicroOpResolver& op_resolver,
    NodeAndRegistration* node_and_registrations) {
  TfLiteStatus status = kTfLiteOk;
  auto* opcodes = model_->operator_codes();
  MicroBuiltinDataAllocator builtin_data_allocator(memory_allocator_);
  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    const auto* op = subgraph_->operators()->Get(i);
    const size_t index = op->opcode_index();
    if (index >= opcodes->size()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Missing registration for opcode_index %d\n", index);
      return kTfLiteError;
    }
    auto* opcode = (*opcodes)[index];
    status =
        GetRegistrationFromOpCode(opcode, op_resolver, error_reporter_,
                                  &(node_and_registrations[i].registration));
    if (status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Failed to get registration from op code %s\n ",
                           EnumNameBuiltinOperator(opcode->builtin_code()));
      return status;
    }
    const auto* registration = node_and_registrations[i].registration;
    if (registration == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_, "Skipping op for opcode_index %d\n",
                           index);
      return kTfLiteError;
    }
    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    const char* custom_data = nullptr;
    size_t custom_data_size = 0;
    unsigned char* builtin_data = nullptr;

    if (op_type == BuiltinOperator_CUSTOM) {
      // Custom Ops may or may not have a non-null custom_options field.
      if (op->custom_options() != nullptr) {
        custom_data =
            reinterpret_cast<const char*>(op->custom_options()->data());
        custom_data_size = op->custom_options()->size();
      }
    } else {
      if (op->custom_options() != nullptr) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Unsupported behavior: found builtin operator %s with custom "
            "options.\n",
            EnumNameBuiltinOperator(op_type));
        return kTfLiteError;
      }

      MicroOpResolver::BuiltinParseFunction parser =
          op_resolver.GetOpDataParser(op_type);
      if (parser == nullptr) {
        TF_LITE_REPORT_ERROR(error_reporter_, "Did not find a parser for %s",
                             EnumNameBuiltinOperator(op_type));

        return kTfLiteError;
      }
      TF_LITE_ENSURE_STATUS(parser(op, op_type, error_reporter_,
                                   &builtin_data_allocator,
                                   (void**)(&builtin_data)));
    }

    // Disregard const qualifier to workaround with existing API.
    TfLiteIntArray* inputs_array = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(op->inputs()));
    TfLiteIntArray* outputs_array = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(op->outputs()));

    TfLiteNode* node = &(node_and_registrations[i].node);
    *node = {};
    node->inputs = inputs_array;
    node->outputs = outputs_array;
    node->builtin_data = reinterpret_cast<void*>(builtin_data);
    node->custom_initial_data = custom_data;
    node->custom_initial_data_size = custom_data_size;
  }

  return kTfLiteOk;
}  // namespace tflite

size_t MicroAllocator::GetTensorsCount() const {
  return context_->tensors_size;
}

size_t MicroAllocator::GetOperatorsCount() const {
  return subgraph_->operators()->size();
}

ErrorReporter* MicroAllocator::error_reporter() { return error_reporter_; }

}  // namespace tflite
