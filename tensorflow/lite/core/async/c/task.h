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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_C_TASK_H_
#define TENSORFLOW_LITE_CORE_ASYNC_C_TASK_H_

#include <stdint.h>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
/// TfLiteExecutionTask API.
///
/// TfLiteExecutionTask stores the information for a specific
/// execution. It includes the mapping from tensors to the buffer handles as
/// well as the synchronization objects.
/// WARNING: This file contains experimental APIs and subject to change.

/// Buffers
/// --------------------------------------------------------------------------
/// If no synchronization type is set, the input data is default to synchronized
/// (i.e. ready when calling InvokeAsync)

/// Sets the buffer handle to the input / output tensor associated with
/// `tensor_signature_name`.
/// `task` and `tensor_signature_name` must not be nullptr.
/// Returns kTfLiteError if the tensor is not found or nullptr args.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteExecutionTaskSetBuffer(
    TfLiteExecutionTask* task, int32_t io_type,
    const char* tensor_signature_name, TfLiteBufferHandle handle);

/// Returns the buffer handle of the input / output tensor associated with
/// `tensor_signature_name`.
/// `task` and `tensor_signature_name` must not be nullptr.
/// Returns kTfLiteNullBufferHandle if the tensor is not found or null input.
TFL_CAPI_EXPORT extern TfLiteBufferHandle TfLiteExecutionTaskGetBufferByName(
    const TfLiteExecutionTask* task, int32_t io_type,
    const char* tensor_signature_name);

/// The same as `TfLiteExecutionTaskGetBufferByName` but takes tensor index
/// instead of the name from signature.
TFL_CAPI_EXPORT extern TfLiteBufferHandle TfLiteExecutionTaskGetBufferByIndex(
    const TfLiteExecutionTask* task, int tensor_index);

/// Synchronizations
/// --------------------------------------------------------------------------
/// Associates synchronization objects to input / output tensors.
///
/// For input tensor, either a nullptr or default sync type
/// `kTfLiteSyncTypeNoSyncObj` means the input is already ready when scheduling
/// the execution. otherwise, the input data will be ready when the underlying
/// sync object signals. The backend is responsible to close the underlying
/// sync object.
/// For output tensor, if the user does not require the backend to return
/// the sync object, it can set the sync type to default
/// `kTfLiteSyncTypeNoSyncObj` or a nullptr TfLiteSynchronization. It means the
/// data is ready when the application calls `Wait` on the given task. Otherwise
/// the backend needs to provide the sync object according to the sync type
/// and it will be signaled when the output data is ready. The underlying output
/// sync object needs to be closed by the application (or some downstream in the
/// pipeline). The backend will be responsible for duplicating the synch
/// if TfLiteSynchronizations are not nullptr for different output tensor
/// produced by the same backend.
///
/// The application needs to maintain the lifetime of the input
/// TfLiteSynchronizations associated with the task during its invocation.
/// TODO(b/191883048): Revisit if we want to bundle the lifetime of sync with
/// the task itself and delete the TfLiteSynchronization in `Finish(task)`.

/// Sets the sync object to the input / output tensor associated with
/// `tensor_signature_name`.
/// `task` and `tensor_signature_name` must not be nullptr.
/// A nullptr `sync` esentially means the tensor data does not need
/// synchronization.
/// `task` does not take the ownership of `sync`.
/// Returns kTfLiteError if the tensor is not found.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteExecutionTaskSetSync(
    TfLiteExecutionTask* task, int32_t io_type,
    const char* tensor_signature_name, TfLiteSynchronization* sync);

/// Returns the sync object of the input / output tensor associated with
/// `tensor_signature_name`.
/// `task` and `tensor_signature_name` must not be nullptr.
/// Returns nullptr if the tensor is not found or null input.
TFL_CAPI_EXPORT extern TfLiteSynchronization* TfLiteExecutionTaskGetSyncByName(
    const TfLiteExecutionTask* task, int32_t io_type,
    const char* tensor_signature_name);

/// The same as `TfLiteExecutionTaskGetSyncByName` but takes tensor index
/// instead of the name from signature.
TFL_CAPI_EXPORT extern TfLiteSynchronization* TfLiteExecutionTaskGetSyncByIndex(
    const TfLiteExecutionTask* task, int tensor_index);

/// Task execution data
/// Backends may store task specific data for executions. This ease the burden
/// for backends to maintain the mapping across different tasks.
TFL_CAPI_EXPORT extern void* TfLiteExecutionTaskGetDelegateExecutionData(
    const TfLiteExecutionTask* task, TfLiteAsyncKernel* kernel);

TFL_CAPI_EXPORT extern void TfLiteExecutionTaskSetDelegateExecutionData(
    const TfLiteExecutionTask* task, TfLiteAsyncKernel* kernel, void* data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_C_TASK_H_
