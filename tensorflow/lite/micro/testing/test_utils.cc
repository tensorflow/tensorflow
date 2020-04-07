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

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace testing {

TfLiteStatus FakeAllocator::AllocatePersistentBuffer(size_t bytes, void** ptr) {
  uint8_t* addr = memory_allocator_->AllocateFromTail(bytes, kBufferAlignment);
  *ptr = addr;
  return kTfLiteOk;
}

TfLiteStatus FakeAllocator::RequestScratchBufferInArena(int node_idx,
                                                        size_t bytes,
                                                        int* buffer_idx) {
  if (scratch_buffers_count_ >= max_scratch_buffers_count_) {
    return kTfLiteError;
  }
  uint8_t* ptr = memory_allocator_->AllocateFromTail(bytes, kBufferAlignment);
  scratch_buffers_[scratch_buffers_count_] = ptr;
  *buffer_idx = scratch_buffers_count_;
  scratch_buffers_count_++;
  return kTfLiteOk;
}

void FakeAllocator::Reset() {
  // Get A fresh memory allocator.
  memory_allocator_ = CreateInPlaceSimpleMemoryAllocator(arena_, arena_size_);
  TFLITE_DCHECK_NE(memory_allocator_, nullptr);

  // Allocate enough space holding pointers to the scrtach buffers.
  scratch_buffers_ =
      reinterpret_cast<uint8_t**>(memory_allocator_->AllocateFromTail(
          sizeof(uint8_t*) * max_scratch_buffers_count_, alignof(uint8_t*)));
  TFLITE_DCHECK_NE(scratch_buffers_, nullptr);

  scratch_buffers_count_ = 0;
}

void* FakeAllocator::GetScratchBuffer(int buffer_idx) {
  if (buffer_idx < 0 || buffer_idx >= scratch_buffers_count_) {
    return nullptr;
  }
  return scratch_buffers_[buffer_idx];
}

TfLiteStatus FakeContextHelper::AllocatePersistentBuffer(TfLiteContext* ctx,
                                                         size_t bytes,
                                                         void** ptr) {
  return reinterpret_cast<FakeContextHelper*>(ctx->impl_)
      ->allocator_->AllocatePersistentBuffer(bytes, ptr);
}

TfLiteStatus FakeContextHelper::RequestScratchBufferInArena(TfLiteContext* ctx,
                                                            size_t bytes,
                                                            int* buffer_idx) {
  FakeContextHelper* helper = reinterpret_cast<FakeContextHelper*>(ctx->impl_);
  // FakeAllocator doesn't do memory reusing so it doesn't need node_idx to
  // calculate the lifetime of the scratch buffer.
  int node_idx = -1;
  return helper->allocator_->RequestScratchBufferInArena(node_idx, bytes,
                                                         buffer_idx);
}

void* FakeContextHelper::GetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  return reinterpret_cast<FakeContextHelper*>(ctx->impl_)
      ->allocator_->GetScratchBuffer(buffer_idx);
}

void FakeContextHelper::ReportOpError(struct TfLiteContext* context,
                                      const char* format, ...) {
  FakeContextHelper* helper = static_cast<FakeContextHelper*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(helper->error_reporter_, format, args);
  va_end(args);
}

namespace {
constexpr size_t kArenaSize = 10000;
constexpr int kMaxScratchBufferCount = 32;
uint8_t arena[kArenaSize];
}  // namespace

// TODO(b/141330728): Move this method elsewhere as part clean up.
void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     ErrorReporter* error_reporter, TfLiteContext* context) {
  // This should be a large enough arena for each test cases.
  static FakeAllocator allocator(arena, kArenaSize, kMaxScratchBufferCount);
  static FakeContextHelper helper(error_reporter, &allocator);
  // Reset the allocator so that it's ready for another test.
  allocator.Reset();

  *context = {};
  context->recommended_num_threads = 1;
  context->tensors_size = tensors_size;
  context->tensors = tensors;
  context->impl_ = static_cast<void*>(&helper);
  context->AllocatePersistentBuffer = helper.AllocatePersistentBuffer;
  context->RequestScratchBufferInArena = helper.RequestScratchBufferInArena;
  context->GetScratchBuffer = helper.GetScratchBuffer;
  context->ReportError = helper.ReportOpError;

  for (int i = 0; i < tensors_size; ++i) {
    if (context->tensors[i].is_variable) {
      ResetVariableTensor(&context->tensors[i]);
    }
  }
}

}  // namespace testing
}  // namespace tflite
