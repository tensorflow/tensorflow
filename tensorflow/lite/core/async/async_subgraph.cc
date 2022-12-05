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
#include "tensorflow/lite/core/async/async_subgraph.h"

#include <vector>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/common.h"
#include "tensorflow/lite/core/async/task_internal.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace async {

Subgraph* AsyncSubgraph::subgraph() const { return subgraph_; }

TfLiteContext* AsyncSubgraph::context() const { return subgraph_->context(); }

TfLiteOpaqueContext* AsyncSubgraph::opaque_context() const {
  return reinterpret_cast<TfLiteOpaqueContext*>(context());
}

TfLiteAsyncKernel* AsyncSubgraph::async_kernel() const {
  if (async_kernel_ == nullptr) {
    auto* node = reinterpret_cast<TfLiteNode*>(opaque_node_);
    async_kernel_ = reinterpret_cast<TfLiteAsyncKernel*>(node->user_data);
  }
  return async_kernel_;
}

AsyncSubgraph::AsyncSubgraph(Subgraph* subgraph) : subgraph_(subgraph) {
  // Currently we only support one delegate and fully delegated subgph.
  if (!IsFullyDelegated()) {
    subgraph->ReportError("Model is no fully delegated by 1 backend.");
    return;
  }
  // TODO(b/191883048): Add/Check delegate flag to indicate kernel support.
  const TfLiteNode& node =
      subgraph->nodes_and_registration()[subgraph_->execution_plan()[0]].first;
  async_kernel_ = reinterpret_cast<TfLiteAsyncKernel*>(node.user_data);
  // TODO(b/191883048): Add AsyncSubgraph as friend class of Subgraph and
  // remove the const cast.
  opaque_node_ =
      reinterpret_cast<TfLiteOpaqueNode*>(const_cast<TfLiteNode*>(&node));
}

bool AsyncSubgraph::IsFullyDelegated() const {
  if (subgraph_->execution_plan().size() != 1) return false;
  const TfLiteNode& node =
      subgraph_->nodes_and_registration()[subgraph_->execution_plan()[0]].first;
  if (node.delegate == nullptr) return false;
  return true;
}

TfLiteStatus AsyncSubgraph::RegisterBuffer(TfLiteIoType io_type,
                                           const TfLiteBackendBuffer* buffer,
                                           const TfLiteAttributeMap* attrs,
                                           TfLiteBufferHandle* handle) {
  if (buffer == nullptr || attrs == nullptr || handle == nullptr ||
      async_kernel() == nullptr) {
    return kTfLiteError;
  }
  *handle = next_buffer_handle_.fetch_add(1, std::memory_order_relaxed);
  return (*async_kernel_->register_buffer)(
      async_kernel_, reinterpret_cast<TfLiteOpaqueContext*>(context()), io_type,
      buffer, attrs, *handle);
}

TfLiteStatus AsyncSubgraph::RegisterBufferSlice(TfLiteBufferHandle buffer_pool,
                                                const TfLiteAttributeMap* attrs,
                                                TfLiteBufferHandle* handle) {
  if (attrs == nullptr || handle == nullptr || async_kernel() == nullptr) {
    return kTfLiteError;
  }
  *handle = next_buffer_handle_.fetch_add(1, std::memory_order_relaxed);
  return (*async_kernel_->register_buffer_slice)(
      async_kernel_, opaque_context(), buffer_pool, attrs, *handle);
}

TfLiteStatus AsyncSubgraph::UnregisterBuffer(TfLiteBufferHandle handle) {
  if (async_kernel() == nullptr) return kTfLiteError;
  return (*async_kernel_->unregister_buffer)(async_kernel_, opaque_context(),
                                             handle);
}

std::vector<const char*> AsyncSubgraph::SupportedBufferTypes(
    TfLiteIoType io_type) const {
  if (async_kernel() == nullptr) return {};
  return (*async_kernel_->supported_buffer_types)(async_kernel_, io_type);
}

std::vector<const char*> AsyncSubgraph::SupportedSynchronizations(
    TfLiteIoType io_type) const {
  if (async_kernel() == nullptr) return {};
  return (*async_kernel_->supported_synchronizations)(async_kernel_, io_type);
}

bool AsyncSubgraph::ReconcileRestrictions(
    int tensor_index, const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  if (user_provided_attributes == nullptr || merged == nullptr ||
      async_kernel() == nullptr) {
    return false;
  }
  return (*async_kernel_->reconcile_restrictions)(
      async_kernel_, opaque_context(), opaque_node_, tensor_index,
      user_provided_attributes, merged, conflict);
}

TfLiteStatus AsyncSubgraph::SetAttributes(int tensor_index,
                                          const TfLiteAttributeMap* attrs) {
  if (attrs == nullptr || async_kernel() == nullptr) {
    return kTfLiteError;
  }
  return (*async_kernel_->set_attributes)(async_kernel_, opaque_context(),
                                          opaque_node_, tensor_index, attrs);
}

TfLiteStatus AsyncSubgraph::Prepare() {
  if (async_kernel() == nullptr) return kTfLiteError;
  return (*async_kernel_->prepare)(async_kernel_, opaque_context(),
                                   opaque_node_);
}

TfLiteExecutionTask* AsyncSubgraph::CreateTask() {
  return new TfLiteExecutionTask;
}

TfLiteStatus AsyncSubgraph::InvokeAsync(TfLiteExecutionTask* task) {
  if (task == nullptr || async_kernel() == nullptr) {
    return kTfLiteError;
  }
  if (task->task->Scheduled()) {
    TFLITE_LOG(tflite::TFLITE_LOG_ERROR,
               "The task has already been scheduled for execution.");
    return kTfLiteError;
  }
  task->task->SetScheduled(true);
  return (*async_kernel_->eval)(async_kernel_, opaque_context(), opaque_node_,
                                task);
}

TfLiteStatus AsyncSubgraph::Wait(TfLiteExecutionTask* task) {
  if (task == nullptr || async_kernel() == nullptr) {
    return kTfLiteError;
  }
  if (!task->task->Scheduled()) {
    // Nothing to wait.
    return kTfLiteOk;
  }
  task->task->SetScheduled(false);
  return (*async_kernel_->wait)(async_kernel_, opaque_context(), task);
}

TfLiteStatus AsyncSubgraph::Finish(TfLiteExecutionTask* task) {
  if (async_kernel() == nullptr) return kTfLiteError;
  if ((*async_kernel_->finish)(async_kernel_, opaque_context(), task) !=
      kTfLiteOk) {
    subgraph_->ReportError("Failed to finish task.");
    return kTfLiteError;
  }
  delete task;
  return kTfLiteOk;
}

}  // namespace async
}  // namespace tflite
