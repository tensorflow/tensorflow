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
#include "tensorflow/contrib/lite/kernels/gemm_support.h"

#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace gemm_support {

struct RefCountedGemmContext {
  gemmlowp::GemmContext* gemm_context_ = nullptr;
  int num_references_ = 0;
};

void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = reinterpret_cast<RefCountedGemmContext*>(context->gemm_context);
  if (ptr == nullptr) {
    ptr = new RefCountedGemmContext;
    ptr->gemm_context_ = new gemmlowp::GemmContext();
    if (context->recommended_num_threads != -1) {
      ptr->gemm_context_->set_max_num_threads(context->recommended_num_threads);
    }
    ptr->num_references_ = 0;
    context->gemm_context = ptr;
  }
  ptr->num_references_++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = reinterpret_cast<RefCountedGemmContext*>(context->gemm_context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references_ == 0) {
    delete ptr->gemm_context_;
    delete ptr;
    context->gemm_context = nullptr;
  }
}

gemmlowp::GemmContext* GetFromContext(TfLiteContext* context) {
  auto* ptr = reinterpret_cast<RefCountedGemmContext*>(context->gemm_context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return ptr->gemm_context_;
}

}  // namespace gemm_support
}  // namespace tflite
