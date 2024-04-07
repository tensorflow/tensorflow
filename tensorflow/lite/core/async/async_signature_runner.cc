/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/async/async_signature_runner.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/async_subgraph.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/task_internal.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace async {

namespace {

// Returns the tensor index of the given signature name.
// `map` is a mapping from tensor signature name to tensor index.
// Return -1 if name is not found in the map or map is nullptr.
int GetIndex(const std::map<std::string, uint32_t>* map, const char* name) {
  if (map == nullptr) return -1;
  const auto& it = map->find(name);
  return it == map->end() ? -1 : it->second;
}

}  // namespace

int AsyncSignatureRunner::GetTensorIndex(TfLiteIoType io_type,
                                         const char* name) const {
  int tensor_index = -1;
  switch (io_type) {
    case kTfLiteIoTypeInput: {
      tensor_index = GetIndex(input_to_index_, name);
      break;
    };
    case kTfLiteIoTypeOutput: {
      tensor_index = GetIndex(output_to_index_, name);
      break;
    }
    default: {
      return false;
    }
  }
  if (tensor_index < 0) {
    subgraph_->ReportError("Signature tensor name %s was not found", name);
  }
  return tensor_index;
}

AsyncSignatureRunner::AsyncSignatureRunner(
    const internal::SignatureDef* signature_def, Subgraph* subgraph)
    : subgraph_(subgraph) {
  async_subgraph_ = std::make_unique<AsyncSubgraph>(subgraph);
  if (signature_def) {
    signature_key_ = signature_def->signature_key;
    input_to_index_ = &signature_def->inputs;
    output_to_index_ = &signature_def->outputs;
    // Collects the list of input and output tensor names.
    for (const auto& it : *input_to_index_) {
      input_names_.push_back(it.first.c_str());
    }
    for (const auto& it : *output_to_index_) {
      output_names_.push_back(it.first.c_str());
    }
  }
}

TfLiteStatus AsyncSignatureRunner::RegisterBuffer(
    TfLiteIoType io_type, const TfLiteBackendBuffer* buffer,
    const TfLiteAttributeMap* attrs, TfLiteBufferHandle* handle) {
  return async_subgraph_->RegisterBuffer(io_type, buffer, attrs, handle);
}

TfLiteStatus AsyncSignatureRunner::RegisterBufferSlice(
    TfLiteBufferHandle buffer_pool, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle* handle) {
  return async_subgraph_->RegisterBufferSlice(buffer_pool, attrs, handle);
}

TfLiteStatus AsyncSignatureRunner::UnregisterBuffer(TfLiteBufferHandle handle) {
  return async_subgraph_->UnregisterBuffer(handle);
}

const std::vector<const char*>& AsyncSignatureRunner::SupportedBufferTypes(
    TfLiteIoType io_type) const {
  return async_subgraph_->SupportedBufferTypes(io_type);
}
const std::vector<const char*>& AsyncSignatureRunner::SupportedSynchronizations(
    TfLiteIoType io_type) const {
  return async_subgraph_->SupportedSynchronizations(io_type);
}

bool AsyncSignatureRunner::ReconcileRestrictions(
    TfLiteIoType io_type, const char* name,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  auto tensor_index = GetTensorIndex(io_type, name);
  return ReconcileRestrictions(tensor_index, user_provided_attributes, merged,
                               conflict);
}

bool AsyncSignatureRunner::ReconcileRestrictions(
    int tensor_index, const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  if (tensor_index < 0) return false;
  return async_subgraph_->ReconcileRestrictions(
      tensor_index, user_provided_attributes, merged, conflict);
}

TfLiteStatus AsyncSignatureRunner::SetAttributes(
    TfLiteIoType io_type, const char* name, const TfLiteAttributeMap* attrs) {
  auto tensor_index = GetTensorIndex(io_type, name);
  return SetAttributes(tensor_index, attrs);
}

TfLiteStatus AsyncSignatureRunner::SetAttributes(
    int tensor_index, const TfLiteAttributeMap* attrs) {
  if (tensor_index < 0) return kTfLiteError;
  return async_subgraph_->SetAttributes(tensor_index, attrs);
}

TfLiteStatus AsyncSignatureRunner::SetBufferAttributes(
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs) {
  return async_subgraph_->SetBufferAttributes(buffer, attrs);
}

TfLiteStatus AsyncSignatureRunner::GetBufferAttributes(
    const TfLiteBackendBuffer* buffer, TfLiteAttributeMap* attrs) {
  return async_subgraph_->GetBufferAttributes(buffer, attrs);
}

TfLiteStatus AsyncSignatureRunner::PrepareBackends() {
  return async_subgraph_->Prepare();
}

TfLiteExecutionTask* AsyncSignatureRunner::CreateTask() {
  auto* task = async_subgraph_->CreateTask();
  task->task->SetInputNameMap(input_to_index_);
  task->task->SetOutputNameMap(output_to_index_);
  return task;
}

TfLiteStatus AsyncSignatureRunner::InvokeAsync(TfLiteExecutionTask* task) {
  return async_subgraph_->InvokeAsync(task);
}

TfLiteStatus AsyncSignatureRunner::Wait(TfLiteExecutionTask* task) {
  return async_subgraph_->Wait(task);
}

TfLiteStatus AsyncSignatureRunner::Finish(TfLiteExecutionTask* task) {
  return async_subgraph_->Finish(task);
}

const TfLiteOpaqueTensor* AsyncSignatureRunner::input_tensor(
    const char* input_name) const {
  if (auto idx = GetTensorIndex(kTfLiteIoTypeInput, input_name); idx >= 0) {
    return reinterpret_cast<const TfLiteOpaqueTensor*>(subgraph_->tensor(idx));
  }
  subgraph_->ReportError("Input name %s was not found", input_name);
  return nullptr;
}

const TfLiteOpaqueTensor* AsyncSignatureRunner::output_tensor(
    const char* output_name) const {
  if (auto idx = GetTensorIndex(kTfLiteIoTypeOutput, output_name); idx >= 0) {
    return reinterpret_cast<const TfLiteOpaqueTensor*>(subgraph_->tensor(idx));
  }
  subgraph_->ReportError("Output name %s was not found", output_name);
  return nullptr;
}

}  // namespace async
}  // namespace tflite
