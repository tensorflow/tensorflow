/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_C_ASYNC_SIGNATURE_RUNNER_H_
#define TENSORFLOW_LITE_CORE_ASYNC_C_ASYNC_SIGNATURE_RUNNER_H_

#include <stdbool.h>
#include <stdint.h>

#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// APIs for asynchronous execution using TFLite AsyncSignatureRunner.
///
/// WARNING: This is an experimental API and subject to change.

/// Opaque TfLiteAsyncSignatureRunner type.
typedef struct TfLiteAsyncSignatureRunner TfLiteAsyncSignatureRunner;

/// Returns a new async signature runner using the provided interpreter and
/// signature key, or nullptr on failure.
///
/// NOTE: `signature_key` is a null-terminated C string that must match the
/// key of a signature in the interpreter's model.
///
/// NOTE: The returned signature runner should be destroyed, by calling
/// TfLiteAsyncSignatureRunnerDelete(), before the interpreter is destroyed.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteAsyncSignatureRunner*
TfLiteInterpreterGetAsyncSignatureRunner(const TfLiteInterpreter* interpreter,
                                         const char* signature_key);

/// Registers a TfLiteBackendBuffer to the backend.
/// `async_signature_runner`, `buffer`, `attrs` and `handle` should be non-null.
/// If the hardware buffer wrapped in `buffer` is successfully registered,
/// `handle` will be filled with a new buffer handle. Caller can use the buffer
/// handle as input / output buffer in `TfLiteExecutionTask`.
/// Returns kTfLiteError if the registration failed.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerRegisterBuffer(
    TfLiteAsyncSignatureRunner* async_signature_runner, TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle* handle);

/// Registers a buffer slice from a previously registered handle `buffer_pool`.
/// `async_signature_runner`, `attrs` and `handle` should be non-null.
/// If the buffer slice described by `attrs` is successfully registered,
/// output `handle` will be filled with a new buffer handle value.
/// NOTE: `attrs` should contain the information about the buffer slice,
/// e.g. offset and size of the size (if applicable).
/// Returns kTfLiteError if the registration failed.
TFL_CAPI_EXPORT extern TfLiteStatus
TfLiteAsyncSignatureRunnerRegisterBufferSlice(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteBufferHandle buffer_pool, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle* handle);

/// Unregisters a hardware buffer object (or buffer slice) with `handle`.
/// Buffer slices should be unregistered before unregistering the buffer pool
/// it belongs to.
/// Returns kTfLiteError if `handle` is not recognized.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerUnregisterBuffer(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteBufferHandle handle);

/// Returns supported platform-specific hardware buffer types.
///
/// Output `types` will be a array of C strings that can be used as the
/// value of `kTfLiteBufferAttrKeyResourceTypeName`.
/// Output `num_types` is the size of the `types` array, and can be used to
/// access elements in `types`.
///
/// NOTE: The lifetime of the returned array is the same as (and depends on) the
/// lifetime of `signature_runner`.
TFL_CAPI_EXPORT extern TfLiteStatus
TfLiteAsyncSignatureRunnerGetSupportedBufferTypes(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteIoType io_type, const char* const** types, size_t* num_types);

/// Returns supported platform-specific synchronization object types.
///
/// Output `types` will be a array of C strings that can be used as the
/// value of `kTfLiteSynchronizationAttrKeyObjectTypeName`.
/// Output `num_types` is the size of the `types` array, and can be used to
/// access elements in `types`.
///
/// NOTE: The lifetime of the returned array is the same as (and depends on) the
/// lifetime of `signature_runner`.
TFL_CAPI_EXPORT extern TfLiteStatus
TfLiteAsyncSignatureRunnerGetSupportedSynchronizationTypes(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteIoType io_type, const char* const** types, size_t* num_types);

/// Reconciles restrictions with the backend for I/O tensor called `name`.
/// The backend will read `user_provided_attributes` and tries to reconcile
/// those attributes. The backend will also populate its own restrictions
/// back to the caller.
/// The merged attributes will be populated to `merged`. For attributes that
/// the backend does not know or not care about, those will also be copied to
/// `merged` attributes.
/// If there's a conflicting attribute, it will be populated to `conflict` if
/// it's provided.
/// `user_provided_attributes` and `merged` should not be nullptr.
/// Returns true if the reconcilation succeeded and there's no
/// conflicting attributes.
TFL_CAPI_EXPORT extern bool TfLiteAsyncSignatureRunnerReconcileRestrictions(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteIoType io_type, const char* name,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict);

/// Reconciles restrictions with the backend for I/O tensor at `tensor_index`.
/// The backend will read `user_provided_attributes` and tries to reconcile
/// those attributes. The backend will also populate its own restrictions
/// back to the caller.
/// The merged attributes will be populated to `merged`. For attributes that
/// the backend does not know or not care about, those will also be copied to
/// `merged` attributes.
/// If there's a conflicting attribute, it will be populated to `conflict` if
/// it's provided.
/// `user_provided_attributes` and `merged` should not be nullptr.
/// Returns true if the reconcilation succeeded and there's no
/// conflicting attributes.
TFL_CAPI_EXPORT extern bool
TfLiteAsyncSignatureRunnerReconcileRestrictionsByIndex(
    const TfLiteAsyncSignatureRunner* async_signature_runner, int tensor_index,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict);

/// Finalizes I/O tensor `name`'s attributes with `attrs`.
/// The attributes will be forwarded to all backend kernels that depends on
/// tensor. Must call `TfLiteAsyncSignatureRunnerPrepareBackends` after setting
/// new attributes.
/// Callers needs to ensure the lifetime of `name` and `attrs` before this
/// function returns, and those may be deallocated afterwards.
/// Returns true if all backends accept the `attrs`.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerSetAttributes(
    TfLiteAsyncSignatureRunner* async_signature_runner, TfLiteIoType io_type,
    const char* name, const TfLiteAttributeMap* attrs);

/// Finalizes I/O tensor at `tensor_index`'s attributes with `attrs`.
/// The attributes will be forwarded to all backend kernels that depends on
/// tensor. Must call `TfLiteAsyncSignatureRunnerPrepareBackends` after setting
/// new attributes.
/// Callers needs to ensure the lifetime of `name` and `attrs` before this
/// function returns, and those may be deallocated afterwards.
/// Returns true if all backends accept the `attrs`.
TFL_CAPI_EXPORT extern TfLiteStatus
TfLiteAsyncSignatureRunnerSetAttributesByIndex(
    TfLiteAsyncSignatureRunner* async_signature_runner, int tensor_index,
    const TfLiteAttributeMap* attrs);

/// Prepares delegate backends for execution.
/// Must be called after `TfLiteAsyncSignatureRunnerSetAttributes` and before
/// `TfLiteAsyncSignatureRunnerCreateTask`.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerPrepareBackends(
    TfLiteAsyncSignatureRunner* async_signature_runner);

/// Creates an execution task for this signature.
/// Must be called after `TfLiteAsyncSignatureRunnerPrepareBackends` otherwise
/// returns nullptr.
/// When creating a task, all intermediate resources will be allocated
/// for this task.
/// Caller owns the returned task and must release it by calling
/// `TfLiteAsyncSignatureRunnerFinish`.
/// Returns nullptr if the task allocation failed.
TFL_CAPI_EXPORT extern TfLiteExecutionTask*
TfLiteAsyncSignatureRunnerCreateTask(
    TfLiteAsyncSignatureRunner* async_signature_runner);

/// Schedules an asynchronous execution with I/O information
/// provided in `task`.
/// `task` should not be nullptr.
///
/// NOTE: For the same `task`,
/// `Wait` and `InvokeAsync` should be called in pairs, unless `Finish(task)` is
/// called and `task` is freed. The application is responsible
/// to call `Wait` after `InvokeAsync` even if all output tensors are associated
/// with synchronizations.
///
/// Returns kTfLiteError if any backend kernels failed to schedule
/// the execution.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerInvokeAsync(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteExecutionTask* task);

/// Blocks and wait for execution tied to `task` to finish.
/// `task` should not be nullptr.
/// Can be called from multiple threads. All calls will block until the
/// task finishes execution.
///
/// NOTE: For the same `task`,
/// `Wait` and `InvokeAsync` should be called in pairs, unless `Finish(task)` is
/// called and `task` is freed. The application is responsible
/// to call `Wait` after `InvokeAsync` even if all output tensors are associated
/// with synchronizations.
/// If `TfLiteAsyncSignatureRunnerWait` is called without a matching call to
/// `TfLiteAsyncSignatureRunnerInvokeAsync`, returns the latest status code (by
/// default `kTfLiteOk`).
///
/// Returns kTfLiteError if any backends failed to finish the execution.
/// If the task is currently idle, it will return its latest status code.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerWait(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteExecutionTask* task);

/// Finishes the task and release all intermediate resources tied to
/// this task. Must be called exactly once for each `task` object.
/// If there's ongoing execution, this will block wait for the execution
/// to finish.
/// `task` should not be nullptr and will be deleted.
/// NOTE: Caller needs to ensure `Finish` is not called concurrently with
/// `InvokeAsync` or `Wait`.
/// Returns kTfLiteError if fails to release the task. The task will be
/// destroyed regardless of error or not.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteAsyncSignatureRunnerFinish(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteExecutionTask* task);

/// Returns the number of input tensors associated with the signature.
TFL_CAPI_EXPORT extern size_t TfLiteAsyncSignatureRunnerGetInputCount(
    const TfLiteAsyncSignatureRunner* async_signature_runner);

/// Returns the (null-terminated) name of the Nth input in a signature, where N
/// is specified as `input_index`.
///
/// NOTE: The lifetime of the returned name is the same as (and depends on) the
/// lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const char* TfLiteAsyncSignatureRunnerGetInputName(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    int32_t input_index);

/// Returns the number of output tensors associated with the signature.
TFL_CAPI_EXPORT extern size_t TfLiteAsyncSignatureRunnerGetOutputCount(
    const TfLiteAsyncSignatureRunner* async_signature_runner);

/// Returns the (null-terminated) name of the Nth output in a signature, where
/// N is specified as `output_index`.
///
/// NOTE: The lifetime of the returned name is the same as (and depends on) the
/// lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const char* TfLiteAsyncSignatureRunnerGetOutputName(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    int32_t output_index);

/// Returns the input tensor metadata identified by `input_name` in the given
/// signature.
/// Returns nullptr if the given name is not valid.
///
/// NOTE: For AsyncSignatureRunner, tensor data are not stored within
/// `TfLiteOpaqueTensors` but in platform-specific hardware buffer objects.
/// This method is only used for accessing the metadata like shape and data type
/// of the input tensors.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const TfLiteOpaqueTensor*
TfLiteAsyncSignatureRunnerGetInputTensor(
    TfLiteAsyncSignatureRunner* async_signature_runner, const char* input_name);

/// Returns the output tensor metadata identified by `output_name` in the given
/// signature.
/// Returns nullptr if the given name is not valid.
///
/// Note: For AsyncSignatureRunner, tensor data are not stored within
/// `TfLiteOpaqueTensors` but in platform-specific hardware buffer objects.
/// This method is only used for accessing the metadata like shape and data type
/// of the output tensors.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `async_signature_runner`.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const TfLiteOpaqueTensor*
TfLiteAsyncSignatureRunnerGetOutputTensor(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    const char* output_name);

/// Destroys the async signature runner.
TFL_CAPI_EXPORT extern void TfLiteAsyncSignatureRunnerDelete(
    TfLiteAsyncSignatureRunner* signature_runner);

/// Returns a pointer to an array of input tensor indices.  The length of the
/// array can be obtained via a call to
/// `TfLiteAsyncSignatureRunnerGetInputCount`.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const int* TfLiteAsyncSignatureRunnerInputTensorIndices(
    const TfLiteAsyncSignatureRunner* async_signature_runner);

/// Returns a pointer to an array of output tensor indices.  The length of the
/// array can be obtained via a call to
/// `TfLiteAsyncSignatureRunnerGetOutputCount`.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const int* TfLiteAsyncSignatureRunnerOutputTensorIndices(
    const TfLiteAsyncSignatureRunner* async_signature_runner);

/// Returns the tensor metadata identified by `index` in the given
/// signature.
/// Returns nullptr if the given index is not valid or out of bound.
///
/// NOTE: For AsyncSignatureRunner, tensor data are not stored within
/// `TfLiteOpaqueTensors` but in platform-specific hardware buffer objects.
/// This method is only used for accessing the metadata like shape and data type
/// of the input tensors.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `async_signature_runner`.
TFL_CAPI_EXPORT extern const TfLiteOpaqueTensor*
TfLiteAsyncSignatureRunnerGetTensor(
    const TfLiteAsyncSignatureRunner* async_signature_runner, int index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_C_ASYNC_SIGNATURE_RUNNER_H_
