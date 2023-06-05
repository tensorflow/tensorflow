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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_C_ASYNC_KERNEL_H_
#define TENSORFLOW_LITE_CORE_ASYNC_C_ASYNC_KERNEL_H_

// TODO(b/270731824): Add full documentation / tests for this header.
// Please reference to tensorflow/lite/core/async/async_kernel_internal.h
// for documentation.

#include <stdbool.h>

#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// APIs for asynchronous delegate kernel.
///
/// WARNING: This is an experimental API and subject to change.

/// Opaque TfLiteAsyncKernel type.
typedef struct TfLiteAsyncKernel TfLiteAsyncKernel;

/// Creates an async kernel to be initialized.
/// `kernel_data` is the arbitrary data for identifying the async kernel itself
/// and can be retrieved using `TfLiteAsyncKernelGetKernelData`.
/// NOTE: TfLiteAsyncKernel does not own `kernel_data` and callers should
/// ensure `kernel_data` out-lives the returned `TfLiteAsyncKernel`.
TFL_CAPI_EXPORT extern TfLiteAsyncKernel* TfLiteAsyncKernelCreate(
    void* kernel_data);

/// Retrieves the kernel data for identifying the async kernel itself.
TFL_CAPI_EXPORT extern void* TfLiteAsyncKernelGetKernelData(
    const TfLiteAsyncKernel* async_kernel);

/// Buffer operations
/// ======================

/// Sets the callback for registering a piece of platform-specific hardware
/// buffer object.
/// `kernel_data` will be the same value supplied by `TfLiteAsyncKernelCreate`.
///
/// `register_buffer`:
/// Registers the buffer to `handle`.
/// `buffer` and `attrs` lifespan is not guaranteed after the function call
/// returns.
/// kernels should save the stored attributes instead of caching the
/// attribute map object itself.
/// `io_type` specifies whether this buffer is used as an input buffer
/// or an output buffer.
/// `attrs` describes the attributes of the buffer object. It's guaranteed to be
/// of kTfLiteBufferAttrMap type and not null. The application must provide
/// `kTfLiteBufferAttrKeyResourceTypeName` attribute. When additional attributes
/// (e.g. padding, size) are provided, the backend is responsible for validating
/// those attributes to be compatible.
/// Once its registered, TfLite runtime will assign and populate `handle` as
/// the buffer handle.
/// The backend will not own the actual buffer object, but the
/// backend can choose to increase the ref count if underlying implementation
/// supports that.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetRegisterBuffer(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*register_buffer)(
        TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
        TfLiteIoType io_type, const TfLiteBackendBuffer* buffer,
        const TfLiteAttributeMap* attrs, TfLiteBufferHandle handle));

/// Sets the callback for registering a buffer slice from previously registered
/// hardware buffer object.
///
/// `register_buffer_slice`:
///  Registers a buffer slice from a previously registered buffer object.
/// `buffer_pool` is the handle of the buffer pool previously registered.
/// `attrs` contains the information of the buffer slice.
/// Once its registered, TfLite runtime will assign and populate `handle` as
/// the buffer handle.
/// NOTE: The backend is responsible to validate the slicing is "valid":
/// * The slicing is not nested from another slice. (i.e. the `buffer_pool` is
///   a handle returned by `RegisterBuffer`.)
/// * The attributes of the slice (e.g. size, offset) is of valid values
///   from the buffer pool.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetRegisterBufferSlice(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*register_buffer_slice)(TfLiteAsyncKernel* async_kernel,
                                          TfLiteOpaqueContext* context,
                                          TfLiteBufferHandle buffer_pool,
                                          const TfLiteAttributeMap* attrs,
                                          TfLiteBufferHandle handle));

/// Sets the callback for unregistering a buffer handle.
///
/// `unregister_buffer`:
/// Unregisters a buffer or a buffer slice.
/// `handle` is a buffer handle previously assigned via register_* calls.
/// If the `handle` is not recognized, returns error.
/// NOTE: Unregistering the buffer does not mean deallocating the buffer object.
/// But the backend need to reduce the ref-count if ref counting is performed
/// during buffer registration calls.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetUnregisterBuffer(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*unregister_buffer)(TfLiteAsyncKernel* async_kernel,
                                      TfLiteOpaqueContext* context,
                                      TfLiteBufferHandle handle));

/// Reconciliation methods
/// =============================

/// Sets the callback for the backend reporting supported hardware buffer object
/// type names.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetSupportedBufferTypes(
    TfLiteAsyncKernel* async_kernel,
    void (*supported_buffer_types)(const TfLiteAsyncKernel* async_kernel,
                                   TfLiteIoType io_type,
                                   const char* const** types, size_t* n_types));

/// Sets the callback for the backend reporting supported synchronization object
/// type names.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetSupportedSynchronizations(
    TfLiteAsyncKernel* async_kernel,
    void (*supported_synchronizations)(const TfLiteAsyncKernel* async_kernel,
                                       TfLiteIoType io_type,
                                       const char* const** types,
                                       size_t* n_types));

/// Sets the callback for the backend to reconcile execution environment
/// attributes (e.g. buffer / synchronization object properties).
///
/// `reconcile_restrictions`:
/// Reconciles buffer or sync attributes for tensor at `tensor_index`.
/// Fills `merged` with reconciled attributes.
/// If `conflict` is provided, conflicting attributes should be provided there.
/// If the type of the `user_provided_attributes` is not recognizable, returns
/// error.
/// If any of the attribute in the `user_provided_attributes` is not
/// recognizable skip this attribute.
/// Returns true if the attribute map type is recognizable and there's no
/// conflicting attribute.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetReconcileRestrictions(
    TfLiteAsyncKernel* async_kernel,
    bool (*reconcile_restrictions)(
        const TfLiteAsyncKernel* async_kernel,
        const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node,
        int tensor_index, const TfLiteAttributeMap* user_provided_attributes,
        TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict));

/// Sets the callback for the backend to set buffer / synchronization
/// attributes.
///
/// `set_attributes`:
/// Sets the input / output buffer / synchronization object attributes.
/// Backend kernel will check the attributes covers all the requirements.
/// A typical workflow is for callers call Reconcile*Restrictions method
/// above to have a merged attribute list, check all restrictions are met
/// and set input / output attribute here.
/// Returns kTfLiteOk if provided `attrs` covers all requirements.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetSetAttributes(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*set_attributes)(TfLiteAsyncKernel* async_kernel,
                                   TfLiteOpaqueContext* context,
                                   TfLiteOpaqueNode* node, int tensor_index,
                                   const TfLiteAttributeMap* attrs));

/// Sets the callback to prepare the kernels using the information from
/// `set_attributes` calls.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetPrepare(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*prepare)(TfLiteAsyncKernel* async_kernel,
                            TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));

/// Execution methods
/// =============================

/// Sets the callback for the backend to schedule an execution.
///
/// `eval`:
/// Schedules an execution with the information provided in task.
/// The application is responsible for filling out buffer and sync mappings
/// to tensors.
/// Backend will set the sync ptr for related tensors if requested.
/// i.e. SetOutputAttributes has sync implementation requested, and
/// the TfLiteSynchronization is not null for the tensor in `task`.
///
/// TfLite runtime guarantees that the task is in ready state (i.e. no
/// un-ended execution for this task).
///
/// Input synchronizations:
/// If the synchronization of a input tensor is `kTfLiteSyncTypeNoSyncObj`
/// type or it's nullptr, it means the data is ready during Eval call.
/// If not, data will be available when the synchronization signals and the
/// backend is responsible for closing the underlying synchronization.
/// The backend is responsible for dedupping the input sync.
///
/// Output synchronizations:
/// If the synchronization type is `kTfLiteSyncTypeNoSyncObj` or is nullptr,
/// the backend does not need to provide synchronization objects to the user.
/// Otherwise, the backend need to provide the sync according to the sync type
/// provided. The underlying sync object will be closed by the app (or
/// downstream components).
/// If there are multiple non-nullptr kTfLiteSynchronization provided for
/// different output tensors, the backend is responsible for duplicating the
/// synchronization.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetEval(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*eval)(TfLiteAsyncKernel* async_kernel,
                         TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                         TfLiteExecutionTask* task));

/// Sets the callback for the backend to wait for a specific execution.
///
/// `wait`:
/// Waits on the execution scheduled using the task to finish.
/// TfLite runtime guarantees that the task has an un-ended execution.
/// Callers should be able to call `Wait` on the same task from multiple
/// threads, and those calls should return the same status (i.e. if the backend
/// failed to successfully wait on the task, all `Wait` to the task should
/// return the same error before a new invocation is scheduled). Returns
/// kTfLiteOk if the task is finished (w/ or w/o blocking).
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetWait(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*wait)(TfLiteAsyncKernel* async_kernel,
                         TfLiteOpaqueContext* context,
                         TfLiteExecutionTask* task));

/// Sets the callback for the backend to finish an execution and release all
/// intermediate resources.
///
/// `finish`:
/// Finishes the task and clean up allocated resources for the task.
/// May block if there's pending executions.
/// This function will be called once and only once for individual task.
/// Returns kTfLiteOk if there's no error. The backend is responsible to
/// clean up task resources regardless there's error or not.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelSetFinish(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*finish)(TfLiteAsyncKernel* async_kernel,
                           TfLiteOpaqueContext* context,
                           TfLiteExecutionTask* task));

/// Releases `kernel`.
/// Does not release `kernel_data`.
TFL_CAPI_EXPORT extern void TfLiteAsyncKernelDelete(TfLiteAsyncKernel* kernel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_ASYNC_C_ASYNC_KERNEL_H_
