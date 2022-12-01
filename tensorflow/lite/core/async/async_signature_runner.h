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

#include <memory>
#include <vector>

#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/async_subgraph.h"
#include "tensorflow/lite/core/async/common.h"
#include "tensorflow/lite/internal/signature_def.h"
#include "tensorflow/lite/signature_runner.h"

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
  // TODO(b/191883048): Move ctor to private and use `Create` function as
  // factory method.
  // Currently we don't have way to expose signature def from interpreter
  // without changes to interpreter.
  //
  // static AsyncSignatureRunner* Create(const TfLiteInterpreter* interpreter,
  //                                     const char* signature_key);
  // WARNING: This is a temporary constructor before we stablize the API.
  // This if for avoiding making intrusive changes to non experimental code.
  // For now, users can construct AsyncSignatureRunner as follows:
  //  std::unique_ptr<tflite::Interpreter> interpreter;
  //  InterpreterBuilder(model, resolver)(&interpreter);
  //  AsyncSignatureRunner runner(interpreter->GetSignatureRunner("func"));
  explicit AsyncSignatureRunner(SignatureRunner* signature_runner);
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
  std::vector<const char*> SupportedBufferTypes(TfLiteIoType io_type) const;

  // Returns a list of names of supported synchronization types.
  std::vector<const char*> SupportedSynchronizations(
      TfLiteIoType io_type) const;

  // Reconciles registrations with all backends depending on I/O tensor `name`
  // if the backend kernel reads or writes the tensor.
  // Merged attributes will be populated to `merged`.
  // If there's a conflict attribute, it's populated to `conflict` if provided.
  // `user_provided_attributes` and `merged` should not be nullptr.
  // Returns true if the reconcilation successes and there's no conflicting
  // attributes.
  bool ReconcileRestrictions(TfLiteIoType io_type, const char* name,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const;

  // Finalizes the attribute for I/O tensor `name` with `attrs`.
  // The attributes will be sent to all backend kernels that depends on tensor.
  // Must call `Prepare` after setting new attributes.
  // Returns true if all backends accept the `attrs`.
  TfLiteStatus SetAttributes(TfLiteIoType io_type, const char* name,
                             const TfLiteAttributeMap* attrs);

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
  //
  // NOTE: `Wait` and `InvokeAsync` should be called in pairs with the same
  // `task`, unless `Finish(task)` is called and task is freed. The application
  // is responsible to call `Wait` after `InvokeAsync` even if all output
  // tensors are associated with synchronizations.
  //
  // Returns kTfLiteError if any backends failed to finish the execution.
  TfLiteStatus Wait(TfLiteExecutionTask* task);

  // Finishes the task and release all intermediate resources tied to
  // this task.
  // If there's ongoing execution, will block wait for the execution
  // to finish.
  // `task` should not be nullptr and will be deleted.
  // Returns kTfLiteError if failes to release the task.
  TfLiteStatus Finish(TfLiteExecutionTask* task);

 private:
  friend class AsyncSignatureRunnerTest;

  int GetTensorIndex(TfLiteIoType io_type, const char* name) const;

  // Not owned.
  const internal::SignatureDef* signature_def_ = nullptr;
  Subgraph* subgraph_ = nullptr;

  // Currently AsyncSubgraph is owned by SignatureRunner. However after
  // we stablize the interface, the async subgraph should be owned by the
  // interpreter and AsyncSignatureRunner won't own any of the subgraphs.
  std::unique_ptr<AsyncSubgraph> async_subgraph_;
};

}  // namespace async
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SIGNATURE_RUNNER_H_
