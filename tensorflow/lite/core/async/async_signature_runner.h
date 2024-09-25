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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SIGNATURE_RUNNER_H_
#define TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SIGNATURE_RUNNER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/async_subgraph.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"

namespace tflite {
namespace async {

// Forward declarations
class AsyncSignatureRunnerTest;

// WARNING: Experimental interface, subject to change
//
// Async version of SignatureRunner class for running TFLite models using
// SignatureDef.
class AsyncSignatureRunner {
 public:
  // Builds the AsyncSignatureRunner given the provided signature_def and
  // subgraph.
  AsyncSignatureRunner(const internal::SignatureDef* signature_def,
                       Subgraph* subgraph);

  // Registers a TfLiteBackendBuffer to backends.
  // The `buffer` will be sent to all backends and TfLite runtime
  // will assign an unique `handle` for backends to recognize the buffer.
  // `io_type` specifies whether the buffer will be used as an input only
  // or it will be used as an output.
  // `buffer`, `attrs`, and `handle` should not be null.
  // The application must provide the buffer type in `attrs`. It can also
  // include additional attributes for the backends to validate (e.g. padding).
  // Returns kTfLiteError is any of the backends failed to register
  // the buffer (e.g. buffer type is not supported).
  TfLiteStatus RegisterBuffer(TfLiteIoType io_type,
                              const TfLiteBackendBuffer* buffer,
                              const TfLiteAttributeMap* attrs,
                              TfLiteBufferHandle* handle);

  // Registers a buffer slice from a previously registered handle `buffer_pool`.
  // `attrs` needs to contain both the information from the buffer pool
  // as well as slice information (offset and size).
  // `attrs` and `handle` should not be nullptr.
  // If the application choose to provide the buffer type in `attrs` it must be
  // identical to the buffer type of the buffer pool provided during
  // RegisterBuffer call.
  // Returns kTfLiteError if the registration failed (e.g. `buffer_pool`
  // not found).
  TfLiteStatus RegisterBufferSlice(TfLiteBufferHandle buffer_pool,
                                   const TfLiteAttributeMap* attrs,
                                   TfLiteBufferHandle* handle);

  // Unregisters a buffer (or buffer slice) with `handle`.
  // Returns kTfLiteError if `handle` is not recognized.
  TfLiteStatus UnregisterBuffer(TfLiteBufferHandle handle);

  // Returns a list of names of supported buffer types.
  const std::vector<const char*>& SupportedBufferTypes(
      TfLiteIoType io_type) const;

  // Returns a list of names of supported synchronization types.
  const std::vector<const char*>& SupportedSynchronizations(
      TfLiteIoType io_type) const;

  // Reconciles registrations with all backends depending on I/O tensor `name`
  // if the backend kernel reads or writes the tensor.
  // Merged attributes will be populated to `merged`.
  // If there's a conflict attribute, it's populated to `conflict` if provided.
  // `user_provided_attributes` and `merged` should not be nullptr.
  // Returns true if the reconciliation successes and there's no conflicting
  // attributes.
  bool ReconcileRestrictions(TfLiteIoType io_type, const char* name,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const;

  // Reconciles registrations with all backends depending on I/O tensor at
  // `tensor_index` if the backend kernel reads or writes the tensor. Merged
  // attributes will be populated to `merged`. If there's a conflict attribute,
  // it's populated to `conflict` if provided. `user_provided_attributes` and
  // `merged` should not be nullptr.
  // Returns true if the reconciliation successes and there's no conflicting
  // attributes.
  bool ReconcileRestrictions(int tensor_index,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const;

  // Finalizes the attribute for I/O tensor `name` with `attrs`.
  // The attributes will be sent to all backend kernels that depends on tensor.
  // Must call `Prepare` after setting new attributes.
  // Returns true if all backends accept the `attrs`.
  TfLiteStatus SetAttributes(TfLiteIoType io_type, const char* name,
                             const TfLiteAttributeMap* attrs);

  // Finalizes the attribute for I/O tensor at `tensor_index` with `attrs`.
  // The attributes will be sent to all backend kernels that depends on tensor.
  // Must call `Prepare` after setting new attributes.
  // Returns true if all backends accept the `attrs`.
  TfLiteStatus SetAttributes(int tensor_index, const TfLiteAttributeMap* attrs);

  // Set the attributes of a specific buffer. Returns
  // kTfLiteDelegateError if the buffer is not registered.
  TfLiteStatus SetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   const TfLiteAttributeMap* attrs);

  // Get the attributes from a specific buffer. Returns
  // kTfLiteDelegateError if the buffer has not been found in the
  // backends.
  TfLiteStatus GetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   TfLiteAttributeMap* attrs);

  // Prepares delegate backends for execution.
  // Must be called after calling `SetAttributes`.
  TfLiteStatus PrepareBackends();

  // Creates an execution task for this subgraph.
  // Must be called after `Prepare`.
  // When creating task, all intermediate resources will be allocated
  // for this task.
  // The task must be released by calling `Finish`.
  TfLiteExecutionTask* CreateTask();

  // Schedules an asynchronous execution with I/O information
  // provided in `task`.
  // `task` should not be nullptr.
  // Returns kTfLiteError if any backend kernels failed to schedule
  // the execution.
  TfLiteStatus InvokeAsync(TfLiteExecutionTask* task);

  // Blocks and wait for execution tied to `task` to finish.
  // `task` should not be nullptr.
  // Can be called from multiple threads. All calls will block until the
  // task finishes execution.
  //
  // NOTE: `Wait` and `InvokeAsync` should be called in pairs with the same
  // `task`, unless `Finish(task)` is called and task is freed. The application
  // is responsible to call `Wait` after `InvokeAsync` even if all output
  // tensors are associated with synchronizations.
  //
  // Returns kTfLiteError if any backends failed to finish the execution.
  // If the task is currently idle, it will return its latest status code.
  TfLiteStatus Wait(TfLiteExecutionTask* task);

  // Finishes the task and release all intermediate resources tied to
  // this task. Must be and only be called once for the same `task` object.
  // If there's ongoing execution, will block wait for the execution
  // to finish.
  // `task` should not be nullptr and will be deleted.
  // NOTE: Caller needs to ensure `Finish` is not called concurrently with
  // `InvokeAsync` or `Wait`.
  // Returns kTfLiteError if failes to release the task. The task will be
  // destroyed regardless of error or not.
  TfLiteStatus Finish(TfLiteExecutionTask* task);

  /// Returns the key for the corresponding signature.
  const std::string& signature_key() { return signature_key_; }

  /// Returns the number of inputs.
  size_t input_size() const { return subgraph_->inputs().size(); }

  /// Returns the number of outputs.
  size_t output_size() const { return subgraph_->outputs().size(); }

  /// Read-only access to list of signature input names.
  /// Returns an empty vector if the model does not have signature.
  const std::vector<const char*>& input_names() { return input_names_; }

  /// Read-only access to list of signature output names.
  /// Returns an empty vector if the model does not have signature.
  const std::vector<const char*>& output_names() { return output_names_; }

  /// Returns the input tensor information identified by 'input_name' in the
  /// given signature. Returns nullptr if the given name is not valid.
  /// Note: The returned `TfLiteTensor` should only be used to retrieve
  /// tensor metadata (dimension, data type, etc.). Tensor data should only be
  /// accessed via hardware buffer directly.
  const TfLiteOpaqueTensor* input_tensor(const char* input_name) const;

  /// Returns the output tensor information identified by 'output_name' in the
  /// given signature. Returns nullptr if the given name is not valid.
  /// Note: The returned `TfLiteTensor` should only be used to retrieve
  /// tensor metadata (dimension, data type, etc.). Tensor data should only be
  /// accessed via hardware buffer directly.
  const TfLiteOpaqueTensor* output_tensor(const char* output_name) const;

  /// Tensor index based accessors.

  /// Read only access to list of input index.
  const std::vector<int>& inputs() const { return subgraph_->inputs(); }

  /// Read only access to list of output index.
  const std::vector<int>& outputs() const { return subgraph_->outputs(); }

  /// Returns the tensor information by tensor index.
  const TfLiteOpaqueTensor* tensor(int tensor_index) const {
    // The following cast is safe only because this code is part of the
    // TF Lite runtime implementation.  Apps using TF Lite should not rely on
    // TfLiteOpaqueTensor and TfLiteTensor being equivalent.
    return reinterpret_cast<const TfLiteOpaqueTensor*>(
        subgraph_->tensor(tensor_index));
  }

 private:
  friend class AsyncSignatureRunnerTest;

  int GetTensorIndex(TfLiteIoType io_type, const char* name) const;

  std::string signature_key_;

  // The list of input tensor names.
  std::vector<const char*> input_names_;
  // The list of output tensor names.
  std::vector<const char*> output_names_;

  // Not owned.
  // If the model does not have signature def, the name maps will be nullptr.
  const std::map<std::string, uint32_t>* input_to_index_ = nullptr;
  const std::map<std::string, uint32_t>* output_to_index_ = nullptr;

  // Not owned.
  Subgraph* subgraph_ = nullptr;

  // Currently AsyncSubgraph is owned by SignatureRunner. However after
  // we stabilize the interface, the async subgraph should be owned by the
  // interpreter and AsyncSignatureRunner won't own any of the subgraphs.
  std::unique_ptr<AsyncSubgraph> async_subgraph_;
};

}  // namespace async
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SIGNATURE_RUNNER_H_
