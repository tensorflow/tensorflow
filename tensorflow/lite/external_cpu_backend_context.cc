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
#include "tensorflow/lite/external_cpu_backend_context.h"

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace {

TfLiteStatus RefreshExternalCpuBackendContext(TfLiteContext* context) {
  auto* const external_context = static_cast<ExternalCpuBackendContext*>(
      context->GetExternalContext(context, kTfLiteCpuBackendContext));
  if (external_context && external_context->internal_backend_context() &&
      context->recommended_num_threads != -1) {
    external_context->internal_backend_context()->SetMaxNumThreads(
        context->recommended_num_threads);
  }
  return kTfLiteOk;
}
}  // namespace

ExternalCpuBackendContext::ExternalCpuBackendContext()
    : internal_backend_context_(nullptr) {
  this->type = kTfLiteCpuBackendContext;
  this->Refresh = RefreshExternalCpuBackendContext;
}

}  // namespace tflite
