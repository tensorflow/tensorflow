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

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/task_internal.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace async {

namespace {

TfLiteAsyncKernel* GetAsyncKernel(TfLiteContext* context,
                                  const TfLiteRegistration& op_reg,
                                  TfLiteNode& node) {
  if (op_reg.registration_external) {
    // The casts here are only safe because this code is part of TFLite
    // runtime. Applications should not rely on TfLiteContext / TfLiteNode being
    // equivalent to TfLiteOpaqueContext / TfLiteOpaqueNode.
    auto* context_ = reinterpret_cast<TfLiteOpaqueContext*>(context);
    auto* node_ = reinterpret_cast<TfLiteOpaqueNode*>(&node);
    if (op_reg.registration_external->async_kernel_with_data) {
      auto user_data = op_reg.registration_external->user_data;
      return op_reg.registration_external->async_kernel_with_data(
          user_data, context_, node_);
    } else if (op_reg.registration_external->async_kernel) {
      return op_reg.registration_external->async_kernel(context_, node_);
    }
  }
  if (op_reg.async_kernel) {
    return op_reg.async_kernel(context, &node);
  }
  return nullptr;
}

}  // namespace

Subgraph* AsyncSubgraph::subgraph() const { return subgraph_; }

TfLiteContext* AsyncSubgraph::context() const { return subgraph_->context(); }

TfLiteOpaqueContext* AsyncSubgraph::opaque_context() const {
  // The casts here are only safe because this code is part of TFLite
  // runtime. Applications should not rely on TfLiteContext
  // being equivalent to TfLiteOpaqueContext.
  return reinterpret_cast<TfLiteOpaqueContext*>(context());
}

TfLiteAsyncKernel* AsyncSubgraph::async_kernel() const { return async_kernel_; }

AsyncSubgraph::AsyncSubgraph(Subgraph* subgraph) : subgraph_(subgraph) {
  // Currently we only support one delegate and fully delegated subgraph.
  if (!IsFullyDelegated()) {
    subgraph->ReportError("Model is not fully delegated by 1 backend.");
    return;
  }

  // Ensured by `IsFullyDelegated`, there's only 1 node in execution plan.
  auto node_index = subgraph_->execution_plan()[0];
  TfLiteNode& node = subgraph_->nodes_and_registration_[node_index].first;
  const TfLiteRegistration& registration =
      subgraph_->nodes_and_registration_[node_index].second;
  async_kernel_ = GetAsyncKernel(context(), registration, node);
  if (!async_kernel_) {
    subgraph->ReportError("Backend does not support asynchronous execution.");
    return;
  }
  // TODO(b/191883048): Add AsyncSubgraph as friend class of Subgraph and
  // remove the const cast.
  opaque_node_ =
      reinterpret_cast<TfLiteOpaqueNode*>(const_cast<TfLiteNode*>(&node));
#define POPULATE_VECTOR(io_type, accessor, dest)                          \
  {                                                                       \
    const char* const* types = nullptr;                                   \
    size_t n_types = 0;                                                   \
    (*async_kernel_->accessor)(async_kernel_, io_type, &types, &n_types); \
    dest[io_type] = std::vector<const char*>(types, types + n_types);     \
  }

  POPULATE_VECTOR(kTfLiteIoTypeInput, supported_buffer_types,
                  supported_buffer_types_);
  POPULATE_VECTOR(kTfLiteIoTypeOutput, supported_buffer_types,
                  supported_buffer_types_);
  POPULATE_VECTOR(kTfLiteIoTypeInput, supported_synchronizations,
                  supported_synchronizations_);
  POPULATE_VECTOR(kTfLiteIoTypeOutput, supported_synchronizations,
                  supported_synchronizations_);
#undef POPULATE_VECTOR
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

const std::vector<const char*>& AsyncSubgraph::SupportedBufferTypes(
    TfLiteIoType io_type) const {
  return supported_buffer_types_.at(io_type);
}

const std::vector<const char*>& AsyncSubgraph::SupportedSynchronizations(
    TfLiteIoType io_type) const {
  return supported_synchronizations_.at(io_type);
}

bool AsyncSubgraph::ReconcileRestrictions(
    int tensor_index, const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  if (user_provided_attributes == nullptr || merged == nullptr ||
      async_kernel() == nullptr) {
    return false;
  }
  if (tensor_index < 0 || tensor_index >= subgraph_->tensors_size()) {
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
  if (tensor_index < 0 || tensor_index >= subgraph_->tensors_size()) {
    return kTfLiteError;
  }
  return (*async_kernel_->set_attributes)(async_kernel_, opaque_context(),
                                          opaque_node_, tensor_index, attrs);
}

TfLiteStatus AsyncSubgraph::SetBufferAttributes(
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs) {
  return (*async_kernel_->set_buffer_attributes)(async_kernel_, buffer, attrs);
}

TfLiteStatus AsyncSubgraph::GetBufferAttributes(
    const TfLiteBackendBuffer* buffer, TfLiteAttributeMap* attrs) {
  return (*async_kernel_->get_buffer_attributes)(async_kernel_, buffer, attrs);
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
  if (task->task->SetScheduled(true)) {
    TFLITE_LOG(tflite::TFLITE_LOG_ERROR,
               "The task has already been scheduled for execution.");
    return kTfLiteError;
  }
  auto ret = (*async_kernel_->eval)(async_kernel_, opaque_context(),
                                    opaque_node_, task);
  task->task->SetStatus(ret);
  return ret;
}

TfLiteStatus AsyncSubgraph::Wait(TfLiteExecutionTask* task) {
  if (task == nullptr || async_kernel() == nullptr) {
    return kTfLiteError;
  }
  if (!task->task->Scheduled()) {
    // Nothing to wait. Returns the previous status code in case multiple
    // threads are waiting for the same task.
    return task->task->Status();
  }
  auto ret = (*async_kernel_->wait)(async_kernel_, opaque_context(), task);
  task->task->SetStatus(ret);
  task->task->SetScheduled(false);
  return ret;
}

TfLiteStatus AsyncSubgraph::Finish(TfLiteExecutionTask* task) {
  if (async_kernel() == nullptr) return kTfLiteError;
  auto ret = (*async_kernel_->finish)(async_kernel_, opaque_context(), task);
  if (ret != kTfLiteOk) {
    subgraph_->ReportError("Failed to finish task.");
  }
  delete task;
  return ret;
}

}  // namespace async
}  // namespace tflite
