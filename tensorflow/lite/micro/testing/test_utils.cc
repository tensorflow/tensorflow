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

#include "tensorflow/lite/micro/testing/test_utils.h"

#include "tensorflow/lite/micro/simple_memory_allocator.h"

namespace tflite {
namespace testing {

namespace {
// TODO(b/141330728): Refactor out of test_utils.cc
// The variables below (and the AllocatePersistentBuffer function) are only
// needed for the kernel tests and benchmarks, i.e. where we do not have an
// interpreter object, and the fully featured MicroAllocator.
// Currently, these need to be sufficient for all the kernel_tests. If that
// becomes problematic, we can investigate allowing the arena_size to be
// specified for each call to PopulatContext.
constexpr size_t kArenaSize = 10000;
uint8_t raw_arena_[kArenaSize];
SimpleMemoryAllocator* simple_memory_allocator_ = nullptr;
constexpr size_t kBufferAlignment = 16;

// We store the pointer to the ith scratch buffer to implement the Request/Get
// ScratchBuffer API for the tests. scratch_buffers_[i] will be the ith scratch
// buffer and will still be allocated from within raw_arena_.
constexpr size_t kNumScratchBuffers = 5;
uint8_t* scratch_buffers_[kNumScratchBuffers];
size_t scratch_buffer_count_ = 0;

// Note that the context parameter in this function is only needed to match the
// signature of TfLiteContext::AllocatePersistentBuffer and isn't needed in the
// implementation because we are assuming a single global
// simple_memory_allocator_
TfLiteStatus AllocatePersistentBuffer(TfLiteContext* context, size_t bytes,
                                      void** ptr) {
  TFLITE_DCHECK(simple_memory_allocator_ != nullptr);
  TFLITE_DCHECK(ptr != nullptr);
  *ptr = simple_memory_allocator_->AllocateFromTail(bytes, kBufferAlignment);
  if (*ptr == nullptr) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus RequestScratchBufferInArena(TfLiteContext* context, size_t bytes,
                                         int* buffer_index) {
  TFLITE_DCHECK(simple_memory_allocator_ != nullptr);
  TFLITE_DCHECK(buffer_index != nullptr);

  if (scratch_buffer_count_ == kNumScratchBuffers) {
    TF_LITE_REPORT_ERROR(
        static_cast<ErrorReporter*>(context->impl_),
        "Exceeded the maximum number of scratch tensors allowed (%d).",
        kNumScratchBuffers);
    return kTfLiteError;
  }

  // For tests, we allocate scratch buffers from the tail and keep them around
  // for the lifetime of model. This means that the arena size in the tests will
  // be more than what we would have if the scratch buffers could share memory.
  scratch_buffers_[scratch_buffer_count_] =
      simple_memory_allocator_->AllocateFromTail(bytes, kBufferAlignment);
  TFLITE_DCHECK(scratch_buffers_[scratch_buffer_count_] != nullptr);

  *buffer_index = scratch_buffer_count_++;
  return kTfLiteOk;
}

void* GetScratchBuffer(TfLiteContext* context, int buffer_index) {
  TFLITE_DCHECK(scratch_buffer_count_ <= kNumScratchBuffers);
  if (buffer_index >= scratch_buffer_count_) {
    return nullptr;
  }
  return scratch_buffers_[buffer_index];
}

}  // namespace

uint8_t F2Q(float value, float min, float max) {
  int32_t result = ZeroPointFromMinMax<uint8_t>(min, max) +
                   (value / ScaleFromMinMax<uint8_t>(min, max)) + 0.5f;
  if (result < std::numeric_limits<uint8_t>::min()) {
    result = std::numeric_limits<uint8_t>::min();
  }
  if (result > std::numeric_limits<uint8_t>::max()) {
    result = std::numeric_limits<uint8_t>::max();
  }
  return result;
}

// Converts a float value into a signed eight-bit quantized value.
int8_t F2QS(float value, float min, float max) {
  return F2Q(value, min, max) + std::numeric_limits<int8_t>::min();
}

int32_t F2Q32(float value, float scale) {
  double quantized = value / scale;
  if (quantized > std::numeric_limits<int32_t>::max()) {
    quantized = std::numeric_limits<int32_t>::max();
  } else if (quantized < std::numeric_limits<int32_t>::min()) {
    quantized = std::numeric_limits<int32_t>::min();
  }
  return static_cast<int>(quantized);
}

// TODO(b/141330728): Move this method elsewhere as part clean up.
void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     ErrorReporter* error_reporter, TfLiteContext* context) {
  simple_memory_allocator_ =
      SimpleMemoryAllocator::Create(error_reporter, raw_arena_, kArenaSize);
  TFLITE_DCHECK(simple_memory_allocator_ != nullptr);
  scratch_buffer_count_ = 0;

  context->tensors_size = tensors_size;
  context->tensors = tensors;
  context->impl_ = static_cast<void*>(error_reporter);
  context->GetExecutionPlan = nullptr;
  context->ResizeTensor = nullptr;
  context->ReportError = ReportOpError;
  context->AddTensors = nullptr;
  context->GetNodeAndRegistration = nullptr;
  context->ReplaceNodeSubsetsWithDelegateKernels = nullptr;
  context->recommended_num_threads = 1;
  context->GetExternalContext = nullptr;
  context->SetExternalContext = nullptr;

  context->AllocatePersistentBuffer = AllocatePersistentBuffer;
  context->RequestScratchBufferInArena = RequestScratchBufferInArena;
  context->GetScratchBuffer = GetScratchBuffer;

  for (int i = 0; i < tensors_size; ++i) {
    if (context->tensors[i].is_variable) {
      ResetVariableTensor(&context->tensors[i]);
    }
  }
}

TfLiteTensor CreateFloatTensor(std::initializer_list<float> data,
                               TfLiteIntArray* dims, bool is_variable) {
  return CreateFloatTensor(data.begin(), dims, is_variable);
}

TfLiteTensor CreateBoolTensor(std::initializer_list<bool> data,
                              TfLiteIntArray* dims, bool is_variable) {
  return CreateBoolTensor(data.begin(), dims, is_variable);
}

TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   float min, float max, bool is_variable) {
  TfLiteTensor result;
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<uint8_t>(min, max),
                   ZeroPointFromMinMax<uint8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.is_variable = false;
  return result;
}

TfLiteTensor CreateQuantizedTensor(std::initializer_list<uint8_t> data,
                                   TfLiteIntArray* dims, float min, float max,
                                   bool is_variable) {
  return CreateQuantizedTensor(data.begin(), dims, min, max, is_variable);
}

TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   float min, float max, bool is_variable) {
  TfLiteTensor result;
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<int8_t>(min, max),
                   ZeroPointFromMinMax<int8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.is_variable = is_variable;
  return result;
}

TfLiteTensor CreateQuantizedTensor(std::initializer_list<int8_t> data,
                                   TfLiteIntArray* dims, float min, float max,
                                   bool is_variable) {
  return CreateQuantizedTensor(data.begin(), dims, min, max, is_variable);
}

TfLiteTensor CreateQuantizedTensor(float* data, uint8_t* quantized_data,
                                   TfLiteIntArray* dims, bool is_variable) {
  TfLiteTensor result;
  SymmetricQuantize(data, dims, quantized_data, &result.params.scale);
  result.data.uint8 = quantized_data;
  result.type = kTfLiteUInt8;
  result.dims = dims;
  result.params.zero_point = 128;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.is_variable = is_variable;
  return result;
}

TfLiteTensor CreateQuantizedTensor(float* data, int8_t* quantized_data,
                                   TfLiteIntArray* dims, bool is_variable) {
  TfLiteTensor result;
  SignedSymmetricQuantize(data, dims, quantized_data, &result.params.scale);
  result.data.int8 = quantized_data;
  result.type = kTfLiteInt8;
  result.dims = dims;
  result.params.zero_point = 0;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.is_variable = is_variable;
  return result;
}

TfLiteTensor CreateQuantizedTensor(float* data, int16_t* quantized_data,
                                   TfLiteIntArray* dims, bool is_variable) {
  TfLiteTensor result;
  SignedSymmetricQuantize(data, dims, quantized_data, &result.params.scale);
  result.data.i16 = quantized_data;
  result.type = kTfLiteInt16;
  result.dims = dims;
  result.params.zero_point = 0;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int16_t);
  result.is_variable = is_variable;
  return result;
}

TfLiteTensor CreateQuantized32Tensor(const int32_t* data, TfLiteIntArray* dims,
                                     float scale, bool is_variable) {
  TfLiteTensor result;
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  result.dims = dims;
  // Quantized int32 tensors always have a zero point of 0, since the range of
  // int32 values is large, and because zero point costs extra cycles during
  // processing.
  result.params = {scale, 0};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  result.is_variable = is_variable;
  return result;
}

TfLiteTensor CreateQuantized32Tensor(std::initializer_list<int32_t> data,
                                     TfLiteIntArray* dims, float scale,
                                     bool is_variable) {
  return CreateQuantized32Tensor(data.begin(), dims, scale, is_variable);
}

}  // namespace testing
}  // namespace tflite
