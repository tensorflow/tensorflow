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
#include "tensorflow/lite/kernels/internal/backends/backend_support.h"

#include <memory>

#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace internal {
namespace backends {
namespace {

struct RefCountedKernelBackendContext : public TfLiteExternalContext {
  std::unique_ptr<KernelBackendContext> backend_context;
  int num_references = 0;
};

RefCountedKernelBackendContext* GetKernelBackendContext(
    TfLiteContext* context) {
  return reinterpret_cast<RefCountedKernelBackendContext*>(
      context->GetExternalContext(context, kTfLiteKernelBackendContext));
}

TfLiteStatus Refresh(TfLiteContext* context) {
  auto* ptr = GetKernelBackendContext(context);
  if (ptr != nullptr) {
    ptr->backend_context->set_max_num_threads_all(
        context->recommended_num_threads);
  }
  return kTfLiteOk;
}

}  // namespace

void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetKernelBackendContext(context);
  if (ptr == nullptr) {
    ptr = new RefCountedKernelBackendContext;
    ptr->type = kTfLiteKernelBackendContext;
    ptr->Refresh = Refresh;
    ptr->backend_context.reset(new KernelBackendContext());
    if (context->recommended_num_threads != -1) {
      ptr->backend_context->set_max_num_threads_all(
          context->recommended_num_threads);
    }
    ptr->num_references = 0;
    context->SetExternalContext(context, kTfLiteKernelBackendContext, ptr);
  }
  ptr->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetKernelBackendContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    delete ptr;
    context->SetExternalContext(context, kTfLiteKernelBackendContext, nullptr);
  }
}

KernelBackendContext* GetFromContext(TfLiteContext* context) {
  auto* ptr = GetKernelBackendContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return ptr->backend_context.get();
}

}  // namespace backends
}  // namespace internal
}  // namespace tflite
