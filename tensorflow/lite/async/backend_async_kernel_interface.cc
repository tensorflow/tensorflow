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
#include "tensorflow/lite/async/backend_async_kernel_interface.h"

#include <vector>

#include "tensorflow/lite/async/c/async_kernel.h"
#include "tensorflow/lite/async/c/types.h"

namespace tflite {
namespace delegates {

namespace internal {
TfLiteStatus RegisterBuffer(TfLiteAsyncKernel* async_kernel,
                            TfLiteOpaqueContext* context, TfLiteIoType io_type,
                            const TfLiteBackendBuffer* buffer,
                            const TfLiteAttributeMap* attrs,
                            TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->RegisterBuffer(context, io_type, buffer, attrs, handle);
}

// Registers a buffer slice from a previously registered memory.
// `attrs` contains the information of the memory, but also additional slice
// information.
TfLiteStatus RegisterBufferSlice(TfLiteAsyncKernel* async_kernel,
                                 TfLiteOpaqueContext* context,
                                 TfLiteBufferHandle buffer,
                                 const TfLiteAttributeMap* attrs,
                                 TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->RegisterBufferSlice(context, buffer, attrs, handle);
}

// Unregisters a buffer or a buffer slice.
TfLiteStatus UnregisterBuffer(TfLiteAsyncKernel* async_kernel,
                              TfLiteOpaqueContext* context,
                              const TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->UnregisterBuffer(context, handle);
}

// Reconciliations
// ===================

// Inspects the buffer / sync implementation types supported by the backend.
void SupportedBufferTypes(const TfLiteAsyncKernel* async_kernel,
                          TfLiteIoType io_type, const char* const** types,
                          size_t* n_types) {
  if (types == nullptr || n_types == nullptr) return;
  const auto& buf_types = reinterpret_cast<const BackendAsyncKernelInterface*>(
                              TfLiteAsyncKernelGetKernelData(async_kernel))
                              ->SupportedBufferTypes(io_type);
  *types = buf_types.data();
  *n_types = buf_types.size();
}

void SupportedSynchronizations(const TfLiteAsyncKernel* async_kernel,
                               TfLiteIoType io_type, const char* const** types,
                               size_t* n_types) {
  if (types == nullptr || n_types == nullptr) return;
  const auto& sync_types = reinterpret_cast<const BackendAsyncKernelInterface*>(
                               TfLiteAsyncKernelGetKernelData(async_kernel))
                               ->SupportedSynchronizations(io_type);
  *types = sync_types.data();
  *n_types = sync_types.size();
}

// Reconciles buffer or sync attributes for tensor at tensor_index.
// Fills `merged` with reconciled attributes.
// If `conflict` is provided, conflicting attributes will be provided there.
// Returns true if there's no conflict.
bool ReconcileRestrictions(const TfLiteAsyncKernel* async_kernel,
                           const TfLiteOpaqueContext* context,
                           const TfLiteOpaqueNode* node, int tensor_index,
                           const TfLiteAttributeMap* user_provided_attributes,
                           TfLiteAttributeMap* merged,
                           TfLiteAttributeMap* conflict) {
  return reinterpret_cast<const BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->ReconcileRestrictions(context, node, tensor_index,
                              user_provided_attributes, merged, conflict);
}
// Sets the input / output buffer / sync attributes.
// Backend kernel will check the input attributes covers all the requirements.
// A typical workflow is for callers call Reconcile*Restrictions method
// above to have a merged attribute list, check all restrictions are met
// and set input / output attribute here.
// Returns TfLiteOk if provided `attrs` covers all requirements.
TfLiteStatus SetAttributes(TfLiteAsyncKernel* async_kernel,
                           TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                           int tensor_index, const TfLiteAttributeMap* attrs) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->SetAttributes(context, node, tensor_index, attrs);
}

// Prepares the kernel using the information from Set[In|Out]putAttributes
// call above.
TfLiteStatus Prepare(TfLiteAsyncKernel* async_kernel,
                     TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Prepare(context, node);
}

// Execution methods
// =============================

// Schedules an execution with the information provided in task.
// The application is responsible for filling out buffer and sync mappings
// to tensors.
// Backend will set the sync ptr for related tensors if requested.
// i.e. SetOutputAttributes has sync implementation requested, and
// the TfLiteSynchronization is not null for the tensor in `task`.
// Returns TfLiteOk if the execution is successfully scheduled.
TfLiteStatus Eval(TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node, TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Eval(context, node, task);
}

// Waits on the execution scheduled using the task to finish.
TfLiteStatus Wait(TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
                  TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Wait(context, task);
}

// Finishes the task and clean up allocated resources for the task.
TfLiteStatus Finish(TfLiteAsyncKernel* async_kernel,
                    TfLiteOpaqueContext* context, TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Finish(context, task);
}

TfLiteStatus SetBufferAttributes(TfLiteAsyncKernel* async_kernel,
                                 const TfLiteBackendBuffer* buffer,
                                 const TfLiteAttributeMap* attrs) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->SetBufferAttributes(buffer, attrs);
}

TfLiteStatus GetBufferAttributes(TfLiteAsyncKernel* async_kernel,
                                 const TfLiteBackendBuffer* buffer,
                                 TfLiteAttributeMap* attrs) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->GetBufferAttributes(buffer, attrs);
}

}  // namespace internal

BackendAsyncKernelInterface::BackendAsyncKernelInterface() {
  kernel_ = TfLiteAsyncKernelCreate(this);
  TfLiteAsyncKernelSetRegisterBuffer(kernel_, internal::RegisterBuffer);
  TfLiteAsyncKernelSetRegisterBufferSlice(kernel_,
                                          internal::RegisterBufferSlice);
  TfLiteAsyncKernelSetUnregisterBuffer(kernel_, internal::UnregisterBuffer);
  TfLiteAsyncKernelSetSupportedBufferTypes(kernel_,
                                           internal::SupportedBufferTypes);
  TfLiteAsyncKernelSetSupportedSynchronizations(
      kernel_, internal::SupportedSynchronizations);
  TfLiteAsyncKernelSetReconcileRestrictions(kernel_,
                                            internal::ReconcileRestrictions);
  TfLiteAsyncKernelSetSetAttributes(kernel_, internal::SetAttributes);
  TfLiteAsyncKernelSetSetBufferAttributes(kernel_,
                                          internal::SetBufferAttributes);
  TfLiteAsyncKernelSetGetBufferAttributes(kernel_,
                                          internal::GetBufferAttributes);
  TfLiteAsyncKernelSetPrepare(kernel_, internal::Prepare);
  TfLiteAsyncKernelSetEval(kernel_, internal::Eval);
  TfLiteAsyncKernelSetWait(kernel_, internal::Wait);
  TfLiteAsyncKernelSetFinish(kernel_, internal::Finish);
}

}  // namespace delegates
}  // namespace tflite
