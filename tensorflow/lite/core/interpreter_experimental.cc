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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/async/async_signature_runner.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/profiling/root_profiler.h"

namespace tflite {

TfLiteStatus Interpreter::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation, int64_t flags) {
  return primary_subgraph().SetCustomAllocationForTensor(tensor_index,
                                                         allocation, flags);
}

TfLiteStatus Interpreter::SetAllocator(TfLiteAllocator* allocator) {
  for (auto& subgraph : subgraphs_) {
    TF_LITE_ENSURE_STATUS(subgraph->SetAllocator(allocator));
  }
  allocator_ = allocator;
  return kTfLiteOk;
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

TfLiteStatus Interpreter::ModifyGraphWithDelegate(
    TfLiteOpaqueDelegateStruct* delegate) {
  return ModifyGraphWithDelegateImpl(
      reinterpret_cast<TfLiteDelegate*>(delegate));
}

namespace {

void InsertSortedUnique(std::vector<int>* indices, int subgraph_index) {
  auto it = std::lower_bound(indices->begin(), indices->end(), subgraph_index);
  if (it == indices->end() || *it != subgraph_index) {
    indices->insert(it, subgraph_index);
  }
}

}  // namespace

TfLiteStatus Interpreter::ModifyGraphWithDelegate(
    TfLiteDelegate* delegate, const std::vector<int>& active_subgraph_indices) {
  if (!active_subgraph_indices.empty()) {
    bool all_valid = true;
    for (int subgraph_index : active_subgraph_indices) {
      if (subgraph_index < 0 ||
          subgraph_index >= static_cast<int>(subgraphs_.size())) {
        all_valid = false;
        break;
      }
    }
    if (all_valid) {
      for (int subgraph_index : active_subgraph_indices) {
        InsertSortedUnique(&delegation_scope_root_subgraph_indices_,
                           subgraph_index);
      }
    }
  }
  return ModifyGraphWithDelegateImpl(delegate, active_subgraph_indices);
}

TfLiteStatus Interpreter::ModifyGraphWithDelegate(
    TfLiteOpaqueDelegateStruct* delegate,
    const std::vector<int>& active_subgraph_indices) {
  return ModifyGraphWithDelegate(reinterpret_cast<TfLiteDelegate*>(delegate),
                                 active_subgraph_indices);
}

TfLiteStatus Interpreter::ModifyGraphWithDelegateForSignature(
    TfLiteDelegate* delegate, const char* signature_key) {
  if (signature_key == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Signature key must not be null.");
    return kTfLiteApplicationError;
  }

  const int root_subgraph_index = GetSubgraphIndexFromSignature(signature_key);
  if (root_subgraph_index < 0 ||
      root_subgraph_index >= static_cast<int>(subgraphs_.size())) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Signature key '%s' not found in model signatures.",
                         signature_key);
    return kTfLiteApplicationError;
  }

  std::vector<int> active_subgraph_indices;
  TF_LITE_ENSURE_STATUS(GetReachableSubgraphIndicesImpl(
      {root_subgraph_index}, &active_subgraph_indices));

  // Record the scope before delegating so that default delegates honor the
  // caller's signature selection even if this delegate fails to apply.
  InsertSortedUnique(&delegation_scope_root_subgraph_indices_,
                     root_subgraph_index);

  return ModifyGraphWithDelegateImpl(delegate, active_subgraph_indices);
}

TfLiteStatus Interpreter::ModifyGraphWithDelegateForSignature(
    TfLiteOpaqueDelegateStruct* delegate, const char* signature_key) {
  return ModifyGraphWithDelegateForSignature(
      reinterpret_cast<TfLiteDelegate*>(delegate), signature_key);
}

TfLiteStatus Interpreter::GetReachableSubgraphIndices(
    const std::vector<int>& root_subgraph_indices,
    std::vector<int>* reachable_subgraph_indices) const {
  return GetReachableSubgraphIndicesImpl(root_subgraph_indices,
                                         reachable_subgraph_indices);
}

bool Interpreter::HasDelegates() { return primary_subgraph().HasDelegates(); }

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  return primary_subgraph().SetBufferHandle(tensor_index, buffer_handle,
                                            delegate);
}

TfLiteStatus Interpreter::SetBufferHandle(TfLiteTensor* tensor,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  return Subgraph::SetBufferHandleImpl(context_, tensor, buffer_handle,
                                       delegate);
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

async::AsyncSignatureRunner* Interpreter::GetAsyncSignatureRunner(
    const char* signature_key_) {
  auto [signature_key, empty_signature_fallback] =
      ReplaceWithPlaceholderSignatureKeyIfNeeded(signature_key_);
  if (!signature_key) {
    return nullptr;
  }

  auto iter = async_signature_runner_map_.find(signature_key);
  if (iter != async_signature_runner_map_.end()) {
    return &(iter->second);
  }

  if (empty_signature_fallback) {
    placeholder_signature_def_ = CreatePlaceholderSignatureDef();
    auto status = async_signature_runner_map_.insert(
        {signature_key,
         async::AsyncSignatureRunner(placeholder_signature_def_.get(),
                                     &primary_subgraph())});
    return &(status.first->second);
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
