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

#include "tensorflow/lite/micro/kernels/kernel_runner.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace tflite {
namespace micro {

namespace {
constexpr size_t kBufferAlignment = 16;
}  // namespace

// TODO(b/161841696): Consider moving away from global arena buffers:
constexpr int KernelRunner::kNumScratchBuffers_;
constexpr int KernelRunner::kKernelRunnerBufferSize_;
uint8_t KernelRunner::kKernelRunnerBuffer_[];

KernelRunner::KernelRunner(const TfLiteRegistration& registration,
                           TfLiteTensor* tensors, int tensors_size,
                           TfLiteIntArray* inputs, TfLiteIntArray* outputs,
                           void* builtin_data)
    : allocator_(SimpleMemoryAllocator::Create(GetMicroErrorReporter(),
                                               kKernelRunnerBuffer_,
                                               kKernelRunnerBufferSize_)),
      registration_(registration),
      tensors_(tensors),
      mock_micro_graph_(allocator_) {
  // Prepare TfLiteContext:
  context_.impl_ = static_cast<void*>(this);
  context_.ReportError = ReportOpError;
  context_.recommended_num_threads = 1;
  context_.GetTensor = GetTensor;
  context_.GetEvalTensor = GetEvalTensor;
  context_.AllocatePersistentBuffer = AllocatePersistentBuffer;
  context_.RequestScratchBufferInArena = RequestScratchBufferInArena;
  context_.GetScratchBuffer = GetScratchBuffer;
  context_.GetExecutionPlan = GetGraph;
  context_.recommended_num_threads = 0;

  // Prepare TfLiteNode:
  node_.inputs = inputs;
  node_.outputs = outputs;
  node_.builtin_data = builtin_data;
}

TfLiteStatus KernelRunner::InitAndPrepare(const char* init_data,
                                          size_t length) {
  if (registration_.init) {
    node_.user_data = registration_.init(&context_, init_data, length);
  }
  if (registration_.prepare) {
    TF_LITE_ENSURE_STATUS(registration_.prepare(&context_, &node_));
  }
  return kTfLiteOk;
}

TfLiteStatus KernelRunner::Invoke() {
  if (registration_.invoke == nullptr) {
    MicroPrintf("TfLiteRegistration missing invoke function pointer!");
    return kTfLiteError;
  }
  return registration_.invoke(&context_, &node_);
}

TfLiteTensor* KernelRunner::GetTensor(const struct TfLiteContext* context,
                                      int tensor_index) {
  TFLITE_DCHECK(context != nullptr);
  KernelRunner* runner = reinterpret_cast<KernelRunner*>(context->impl_);
  TFLITE_DCHECK(runner != nullptr);

  return &runner->tensors_[tensor_index];
}

TfLiteEvalTensor* KernelRunner::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_index) {
  TFLITE_DCHECK(context != nullptr);
  KernelRunner* runner = reinterpret_cast<KernelRunner*>(context->impl_);
  TFLITE_DCHECK(runner != nullptr);

  TfLiteEvalTensor* eval_tensor =
      reinterpret_cast<TfLiteEvalTensor*>(runner->allocator_->AllocateTemp(
          sizeof(TfLiteEvalTensor), alignof(TfLiteEvalTensor)));
  TFLITE_DCHECK(eval_tensor != nullptr);

  // In unit tests, the TfLiteTensor pointer contains the source of truth for
  // buffers and values:
  eval_tensor->data = runner->tensors_[tensor_index].data;
  eval_tensor->dims = runner->tensors_[tensor_index].dims;
  eval_tensor->type = runner->tensors_[tensor_index].type;
  return eval_tensor;
}

void* KernelRunner::AllocatePersistentBuffer(TfLiteContext* context,
                                             size_t bytes) {
  TFLITE_DCHECK(context != nullptr);
  KernelRunner* runner = reinterpret_cast<KernelRunner*>(context->impl_);
  TFLITE_DCHECK(runner != nullptr);

  return runner->allocator_->AllocateFromTail(bytes, kBufferAlignment);
}

TfLiteStatus KernelRunner::RequestScratchBufferInArena(TfLiteContext* context,
                                                       size_t bytes,
                                                       int* buffer_index) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(buffer_index != nullptr);

  KernelRunner* runner = reinterpret_cast<KernelRunner*>(context->impl_);
  TFLITE_DCHECK(runner != nullptr);

  if (runner->scratch_buffer_count_ == kNumScratchBuffers_) {
    MicroPrintf("Exceeded the maximum number of scratch tensors allowed (%d).",
                kNumScratchBuffers_);
    return kTfLiteError;
  }

  // For tests, we allocate scratch buffers from the tail and keep them around
  // for the lifetime of model. This means that the arena size in the tests will
  // be more than what we would have if the scratch buffers could share memory.
  runner->scratch_buffers_[runner->scratch_buffer_count_] =
      runner->allocator_->AllocateFromTail(bytes, kBufferAlignment);
  TFLITE_DCHECK(runner->scratch_buffers_[runner->scratch_buffer_count_] !=
                nullptr);

  *buffer_index = runner->scratch_buffer_count_++;
  return kTfLiteOk;
}

void* KernelRunner::GetScratchBuffer(TfLiteContext* context, int buffer_index) {
  TFLITE_DCHECK(context != nullptr);
  KernelRunner* runner = reinterpret_cast<KernelRunner*>(context->impl_);
  TFLITE_DCHECK(runner != nullptr);

  TFLITE_DCHECK(runner->scratch_buffer_count_ <= kNumScratchBuffers_);
  if (buffer_index >= runner->scratch_buffer_count_) {
    return nullptr;
  }
  return runner->scratch_buffers_[buffer_index];
}

void KernelRunner::ReportOpError(struct TfLiteContext* context,
                                 const char* format, ...) {
  va_list args;
  va_start(args, format);
  GetMicroErrorReporter()->Report(format, args);
  va_end(args);
}

TfLiteStatus KernelRunner::GetGraph(struct TfLiteContext* context,
                                    TfLiteIntArray** args) {
  TFLITE_DCHECK(context != nullptr);
  KernelRunner* runner = reinterpret_cast<KernelRunner*>(context->impl_);
  TFLITE_DCHECK(runner != nullptr);
  // TODO(b/188226309): Design a cleaner way to get a graph from kernel context.
  *args = reinterpret_cast<TfLiteIntArray*>(runner->GetMockGraph());
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
