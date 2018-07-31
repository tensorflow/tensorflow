/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_GEMM_SUPPORT_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_GEMM_SUPPORT_H_

#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/context.h"

namespace tflite {
namespace gemm_support {

// Returns the GemmContext stored in 'context', allowing multiple ops to
// share a single object, as long as they share a TfLiteContext. The caller
// must ensure that this is called between IncrementUsageCounter() and
// DecrementUsageCounter(). For example, in the implementation of an op:
//   void* Init(TfLiteContext* context, const char*, size_t) {
//     gemm_support::IncrementUsageCounter(context);
//     return nullptr;
//   }
//   void Free(TfLiteContext* context, void*) {
//     gemm_support::DecrementUsageCounter(context);
//   }
//   TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
//     auto* gemm_context = gemm_support::GetFromContext(context);
//   }
gemmlowp::GemmContext* GetFromContext(TfLiteContext* context);

// Let the framework know that the GemmContext stored in 'context' will be used
// by an op. If necessary a new GemmContext is created and placed in 'context'.
void IncrementUsageCounter(TfLiteContext* context);

// Let the framework know that the op stopped using the GemmContext stored in
// 'context'. If there are no more usages the GemmContext will be deleted.
void DecrementUsageCounter(TfLiteContext* context);

}  // namespace gemm_support
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_GEMM_SUPPORT_H_
