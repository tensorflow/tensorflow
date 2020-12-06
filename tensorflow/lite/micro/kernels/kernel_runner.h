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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_RUNNER_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_RUNNER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"

namespace tflite {
namespace micro {

// Helper class to perform a simulated kernel (i.e. TfLiteRegistration)
// lifecycle (init, prepare, invoke). All internal allocations are handled by
// this class. Simply pass in the registration, list of required tensors, inputs
// array, outputs array, and any pre-builtin data. Calling Invoke() will
// automatically walk the kernel and outputs will be ready on the TfLiteTensor
// output provided during construction.
class KernelRunner {
 public:
  KernelRunner(const TfLiteRegistration& registration, TfLiteTensor* tensors,
               int tensors_size, TfLiteIntArray* inputs,
               TfLiteIntArray* outputs, void* builtin_data,
               ErrorReporter* error_reporter);

  // Calls init and prepare on the kernel (i.e. TfLiteRegistration) struct. Any
  // exceptions will be reported through the error_reporter and returned as a
  // status code here.
  TfLiteStatus InitAndPrepare(const char* init_data = nullptr,
                              size_t length = 0);

  // Calls init, prepare, and invoke on a given TfLiteRegistration pointer.
  // After successful invoke, results will be available in the output tensor as
  // passed into the constructor of this class.
  TfLiteStatus Invoke();

 protected:
  static TfLiteTensor* GetTensor(const struct TfLiteContext* context,
                                 int tensor_index);
  static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                         int tensor_index);
  static void* AllocatePersistentBuffer(TfLiteContext* context, size_t bytes);
  static TfLiteStatus RequestScratchBufferInArena(TfLiteContext* context,
                                                  size_t bytes,
                                                  int* buffer_index);
  static void* GetScratchBuffer(TfLiteContext* context, int buffer_index);
  static void ReportOpError(struct TfLiteContext* context, const char* format,
                            ...);

 private:
  static constexpr int kNumScratchBuffers_ = 12;

  static constexpr int kKernelRunnerBufferSize_ = 10000;
  static uint8_t kKernelRunnerBuffer_[kKernelRunnerBufferSize_];

  SimpleMemoryAllocator* allocator_ = nullptr;
  const TfLiteRegistration& registration_;
  TfLiteTensor* tensors_ = nullptr;
  ErrorReporter* error_reporter_ = nullptr;

  TfLiteContext context_ = {};
  TfLiteNode node_ = {};

  int scratch_buffer_count_ = 0;
  uint8_t* scratch_buffers_[kNumScratchBuffers_];
};

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_RUNNER_H_
