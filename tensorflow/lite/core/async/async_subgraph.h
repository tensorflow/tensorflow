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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SUBGRAPH_H_
#define TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SUBGRAPH_H_

#include <atomic>
#include <map>
#include <vector>

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace async {

// Forward declaration
class AsyncSubgraphTestPeer;

// AsyncSubgraph class manages to dispatch I/O information and
// schedule executions to underlying delegate kernels.
// TODO(b/191883048): Currently we require either `AllocateTensors` or
// `EnsureTensorAllocation` called to ensure the backend kernels are prepared.
// However, we don't need to allocate the CPU memory for input / output tensors.
// We need customize the OpPrepare or memory planner to skip the allocation
// for user provided buffer case.
class AsyncSubgraph {
 public:
  explicit AsyncSubgraph(Subgraph* subgraph);

  // Returns the underlying TfLite subgraph.
  Subgraph* subgraph() const;

  // Returns the TfLiteContext of the subgraph.
  TfLiteContext* context() const;

  // Registers a TfLiteBackendBuffer to backends.
  // The `buffer` will be sent to all backends and TfLite runtime
  // will assign an unique `handle` for backends to recognize the buffer.
  // `buffer`, `attrs`, and `handle` should not be null.
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
  //
  // NOTE: When using sliced buffer as output buffer, the application needs to
  // make sure slices from the same buffer pool should not be used across
  // different executions (from InvokeAsync call to the output sync signals)
  // otherwise data corruption may occur.
  // TODO(b/243175542): Programmatically ensure slices from one buffer are used
  // exclusively by one backend to write to for a single execution.
  //
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

  // Reconciles registrations with all backends depending on tensor at
  // `tensor_index` if the backend kernel reads or writes the tensor.
  // Merged attributes will be populated to `merged`.
  // If there's a conflict attribute, it's populated to `conflict` if provided.
  // `user_provided_attributes` and `merged` should not be nullptr.
  // Returns true if the reconcilation successes and there's no conflicting
  // attributes.
  bool ReconcileRestrictions(int tensor_index,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const;

  // Finalizes the attribute for tensor at `tensor_index` with `attrs`.
  // The attributes will be sent to all backend kernels that depends on tensor
  // at `tensor_index`.
  // Must call `Prepare` after setting new attributes.
  // Returns true if all backends accept the `attrs`.
  TfLiteStatus SetAttributes(int tensor_index, const TfLiteAttributeMap* attrs);

  // Prepares delegate backends for execution.
  // Must be called after calling `SetAttributes`.
  TfLiteStatus Prepare();

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
  // Returns kTfLiteError if any backends failed to finish the execution.
  // If the task is currently idle, it will return its latest status code.
  TfLiteStatus Wait(TfLiteExecutionTask* task);

  // Finishes the task and release all intermediate resources tied to
  // this task.
  // If there's ongoing execution, will block wait for the execution
  // to finish.
  // `task` should not be nullptr and will be deleted.
  // Returns kTfLiteError if failes to release the task. In this case `task`
  // will not be deleted.
  TfLiteStatus Finish(TfLiteExecutionTask* task);

 private:
  friend class AsyncSubgraphTestPeer;

  // Returns true if the subgraph is fully delegated by 1 backend.
  bool IsFullyDelegated() const;

  // Returns the opaque TfLiteContext of the subgraph.
  TfLiteOpaqueContext* opaque_context() const;

  // Returns the async backend kernel that delegates the subgraph.
  // NOTE: Since we assume only 1 backend will delegate the model, we cache
  // the async kernel instance. In theory, the subgraph should iterate through
  // execution plan to fetch the individual async kernels and operate
  // respectively.
  TfLiteAsyncKernel* async_kernel() const;

  // Not owned.
  Subgraph* subgraph_ = nullptr;

  // Next buffer handle to assign in Register* calls.
  std::atomic<TfLiteBufferHandle> next_buffer_handle_ = {0};

  // Supported buffer and sync types.
  std::map<TfLiteIoType, std::vector<const char*>> supported_buffer_types_;
  std::map<TfLiteIoType, std::vector<const char*>> supported_synchronizations_;

  // Currently AsyncSubgraph only support fully delegated by 1 backend case.
  // Not owned.
  mutable TfLiteAsyncKernel* async_kernel_ = nullptr;
  TfLiteOpaqueNode* opaque_node_ = nullptr;
};

}  // namespace async
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_ASYNC_SUBGRAPH_H_
