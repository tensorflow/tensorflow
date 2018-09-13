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

#include <memory>

#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace gemm_support {
namespace {

struct RefCountedGemmContext : public TfLiteExternalContext {
  std::unique_ptr<gemmlowp::GemmContext> gemm_context;
  int num_references = 0;
};

RefCountedGemmContext* GetGemmLowpContext(TfLiteContext* context) {
  return reinterpret_cast<RefCountedGemmContext*>(
      context->GetExternalContext(context, kTfLiteGemmLowpContext));
}

TfLiteStatus Refresh(TfLiteContext* context) {
  auto* ptr = GetGemmLowpContext(context);
  if (ptr != nullptr) {
    ptr->gemm_context->set_max_num_threads(context->recommended_num_threads);
  }
  return kTfLiteOk;
}

}  // namespace

void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetGemmLowpContext(context);
  if (ptr == nullptr) {
    ptr = new RefCountedGemmContext;
    ptr->type = kTfLiteGemmLowpContext;
    ptr->Refresh = Refresh;
    ptr->gemm_context.reset(new gemmlowp::GemmContext());
    if (context->recommended_num_threads != -1) {
      ptr->gemm_context->set_max_num_threads(context->recommended_num_threads);
    }
    ptr->num_references = 0;
    context->SetExternalContext(context, kTfLiteGemmLowpContext, ptr);
  }
  ptr->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetGemmLowpContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    delete ptr;
    context->SetExternalContext(context, kTfLiteGemmLowpContext, nullptr);
  }
}

gemmlowp::GemmContext* GetFromContext(TfLiteContext* context) {
  auto* ptr = GetGemmLowpContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return ptr->gemm_context.get();
}

}  // namespace gemm_support
}  // namespace tflite
