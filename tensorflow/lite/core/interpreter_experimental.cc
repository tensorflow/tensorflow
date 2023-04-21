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

#include <stddef.h>
#include <stdlib.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ruy/denormal.h"  // from @ruy
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/async/async_signature_runner.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/util.h"

namespace tflite {

TfLiteStatus Interpreter::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation, int64_t flags) {
  return primary_subgraph().SetCustomAllocationForTensor(tensor_index,
                                                         allocation, flags);
}

TfLiteStatus Interpreter::ReleaseNonPersistentMemory() {
  // TODO(b/138790287): We could do this for all subgraphs whose tensors have
  // been allocated. However, AllocateTensors() relies on Control Flow ops to
  // allocate tensors on 'children' subgraphs. Revisit this if required.
  return primary_subgraph().ReleaseNonPersistentMemory();
}

TfLiteStatus Interpreter::ResetVariableTensors() {
  for (auto& subgraph : subgraphs_) {
    TF_LITE_ENSURE_STATUS(subgraph->ResetVariableTensors());
  }
  return kTfLiteOk;
}

void Interpreter::SetAllowFp16PrecisionForFp32(bool allow) {
  for (auto& subgraph : subgraphs_) {
    subgraph->context()->allow_fp32_relax_to_fp16 = allow;
  }
}

// TODO(b/121264966): Subgraphs added after cancellation is set will not get the
// cancellation function added to their context.
void Interpreter::SetCancellationFunction(void* data,
                                          bool (*check_cancelled_func)(void*)) {
  for (auto& subgraph : subgraphs_) {
    subgraph->SetCancellationFunction(data, check_cancelled_func);
  }
}

bool Interpreter::IsCancelled() { return primary_subgraph().IsCancelled(); }

TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  return ModifyGraphWithDelegateImpl(delegate);
}

bool Interpreter::HasDelegates() { return primary_subgraph().HasDelegates(); }

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  TfLiteTensor* tensor = primary_subgraph().tensor(tensor_index);

  TF_LITE_ENSURE(context_,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE_STATUS(TfLiteDelegateFreeBufferHandleInternal(
        context_, tensor->delegate, &(tensor->buffer_handle)));
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  TfLiteTensor* tensor = primary_subgraph().tensor(tensor_index);

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

void Interpreter::SetProfiler(Profiler* profiler) {
  if (profiler == nullptr) {
    root_profiler_ = nullptr;
    return;
  }
  if (root_profiler_ != nullptr) root_profiler_->RemoveChildProfilers();
  AddProfiler(profiler);
}

void Interpreter::SetProfiler(std::unique_ptr<Profiler> profiler) {
  SetProfilerImpl(std::move(profiler));
}

void Interpreter::AddProfiler(Profiler* profiler) {
  if (profiler == nullptr) return;
  if (root_profiler_ == nullptr) {
    root_profiler_ = std::make_unique<profiling::RootProfiler>();
  }
  root_profiler_->AddProfiler(profiler);
  SetSubgraphProfiler();
}

Profiler* Interpreter::GetProfiler() {
  return primary_subgraph().GetProfiler();
}

TfLiteStatus Interpreter::ApplyOptions(InterpreterOptions* options) {
  return ApplyOptionsImpl(options);
}

SignatureRunner* Interpreter::GetSignatureRunner(const char* signature_key) {
  auto iter = signature_runner_map_.find(signature_key);
  if (iter != signature_runner_map_.end()) {
    return &(iter->second);
  }

  // Default delegates are applied once for all subgraphs. Only returns error
  // when the status is kTfLiteError. For other statuses, it will fall back to
  // the default implementation.
  if (ApplyLazyDelegateProviders() == kTfLiteError) {
    return nullptr;
  }

  for (const auto& signature : signature_defs_) {
    if (signature.signature_key == signature_key) {
      auto status = signature_runner_map_.insert(
          {signature_key,
           SignatureRunner(&signature, subgraph(signature.subgraph_index))});
      return &(status.first->second);
    }
  }
  return nullptr;
}

async::AsyncSignatureRunner* Interpreter::GetAsyncSignatureRunner(
    const char* signature_key) {
  auto iter = async_signature_runner_map_.find(signature_key);
  if (iter != async_signature_runner_map_.end()) {
    return &(iter->second);
  }

  for (const auto& signature : signature_defs_) {
    if (signature.signature_key == signature_key) {
      auto status = async_signature_runner_map_.insert(
          {signature_key, async::AsyncSignatureRunner(
                              &signature, subgraph(signature.subgraph_index))});
      return &(status.first->second);
    }
  }

  return nullptr;
}

}  // namespace tflite
