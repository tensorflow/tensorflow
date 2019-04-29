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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_SUPPORT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_SUPPORT_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/backends/backend_context.h"

namespace tflite {
namespace internal {
namespace backends {
// Returns the BackendKernelContext stored in 'context', allowing multiple ops
// to share a single object, as long as they share a TfLiteContext. The caller
// must ensure that this is called between IncrementUsageCounter() and
// DecrementUsageCounter(). For example, in the implementation of an op:
//   void* Init(TfLiteContext* context, const char*, size_t) {
//     tflite::ops::backends::IncrementUsageCounter(context);
//     return nullptr;
//   }
//   void Free(TfLiteContext* context, void*) {
//     tflite::ops::backends::DecrementUsageCounter(context);
//   }
//   TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
//     auto* backends_context =
//     tflite::ops::backends::::GetFromContext(context);
//   }
KernelBackendContext* GetFromContext(TfLiteContext* context);

// Let the framework know that the BackendKernelContext stored in 'context' will
// be used by an op. If necessary a new BackendKernelContext is created and
// placed in 'context'.
void IncrementUsageCounter(TfLiteContext* context);

// Let the framework know that the op stopped using the BackendKernelContext
// stored in 'context'. If there are no more usages the BackendKernelContext
// will be deleted.
void DecrementUsageCounter(TfLiteContext* context);

}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_SUPPORT_H_
