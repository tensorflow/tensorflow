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
#include "tensorflow/lite/kernels/cpu_backend_support.h"

#include <memory>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace cpu_backend_support {

namespace {

// TODO(b/130950871) we probably shouldn't be using any reference-counting
// but this is an existing idiom.
struct RefCountedCpuBackendContext : public TfLiteExternalContext {
  std::unique_ptr<CpuBackendContext> cpu_backend_context;
  int num_references = 0;
};

RefCountedCpuBackendContext* GetCpuBackendContext(TfLiteContext* context) {
  return static_cast<RefCountedCpuBackendContext*>(
      context->GetExternalContext(context, kTfLiteCpuBackendContext));
}

TfLiteStatus Refresh(TfLiteContext* context) {
  auto* refcounted = GetCpuBackendContext(context);
  if (refcounted != nullptr) {
    refcounted->cpu_backend_context->set_max_num_threads(
        context->recommended_num_threads);
  }
  return kTfLiteOk;
}

}  // namespace

void IncrementUsageCounter(TfLiteContext* context) {
  RefCountedCpuBackendContext* refcounted = GetCpuBackendContext(context);
  if (refcounted == nullptr) {
    refcounted = new RefCountedCpuBackendContext;
    refcounted->type = kTfLiteCpuBackendContext;
    refcounted->Refresh = Refresh;
    refcounted->cpu_backend_context.reset(new CpuBackendContext);
    if (context->recommended_num_threads != -1) {
      refcounted->cpu_backend_context->set_max_num_threads(
          context->recommended_num_threads);
    }
    refcounted->num_references = 0;
    context->SetExternalContext(context, kTfLiteCpuBackendContext, refcounted);
  }
  refcounted->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  RefCountedCpuBackendContext* refcounted = GetCpuBackendContext(context);
  if (refcounted == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--refcounted->num_references == 0) {
    delete refcounted;
    context->SetExternalContext(context, kTfLiteCpuBackendContext, nullptr);
  }
}

CpuBackendContext* GetFromContext(TfLiteContext* context) {
  RefCountedCpuBackendContext* refcounted = GetCpuBackendContext(context);
  if (refcounted == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return refcounted->cpu_backend_context.get();
}

}  // namespace cpu_backend_support
}  // namespace tflite
