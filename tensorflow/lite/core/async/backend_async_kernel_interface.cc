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
#include "tensorflow/lite/core/async/backend_async_kernel_interface.h"

#include <vector>

namespace tflite {
namespace delegates {

namespace internal {
TfLiteStatus RegisterBuffer(TfLiteAsyncKernel* async_kernel,
                            TfLiteOpaqueContext* context, int32_t io_type,
                            const TfLiteBackendBuffer* buffer,
                            const TfLiteAttributeMap* attrs,
                            TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
      ->RegisterBuffer(context, static_cast<TfLiteIoType>(io_type), buffer,
                       attrs, handle);
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
             async_kernel->kernel_data)
      ->RegisterBufferSlice(context, buffer, attrs, handle);
}

// Unregisters a buffer or a buffer slice.
TfLiteStatus UnregisterBuffer(TfLiteAsyncKernel* async_kernel,
                              TfLiteOpaqueContext* context,
                              const TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
      ->UnregisterBuffer(context, handle);
}

// Reconciliations
// ===================

// Inspects the buffer / sync implementation types supported by the backend.
std::vector<const char*> SupportedBufferTypes(
    const TfLiteAsyncKernel* async_kernel, int32_t io_type) {
  return reinterpret_cast<const BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
      ->SupportedBufferTypes(static_cast<TfLiteIoType>(io_type));
}
std::vector<const char*> SupportedSynchronizations(
    const TfLiteAsyncKernel* async_kernel, int32_t io_type) {
  return reinterpret_cast<const BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
      ->SupportedSynchronizations(static_cast<TfLiteIoType>(io_type));
}

// Reconciles buffer or sync attributes for tensor at tensor_index.
// Fills `merged` with reconciled attributes.
// If `conflict` is provided, conflicting attributes will be provided there.
// Returns true if there's no conflict.
bool ReconcileRestrictions(const TfLiteAsyncKernel* async_kernel,
                           TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                           int tensor_index,
                           const TfLiteAttributeMap* user_provided_attributes,
                           TfLiteAttributeMap* merged,
                           TfLiteAttributeMap* conflict) {
  return reinterpret_cast<const BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
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
             async_kernel->kernel_data)
      ->SetAttributes(context, node, tensor_index, attrs);
}

// Prepares the kernel using the information from Set[In|Out]putAttributes
// call above.
TfLiteStatus Prepare(TfLiteAsyncKernel* async_kernel,
                     TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
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
             async_kernel->kernel_data)
      ->Eval(context, node, task);
}

// Waits on the execution scheduled using the task to finish.
TfLiteStatus Wait(TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
                  TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
      ->Wait(context, task);
}

// Finishes the task and clean up allocated resources for the task.
TfLiteStatus Finish(TfLiteAsyncKernel* async_kernel,
                    TfLiteOpaqueContext* context, TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             async_kernel->kernel_data)
      ->Finish(context, task);
}

}  // namespace internal

BackendAsyncKernelInterface::BackendAsyncKernelInterface() {
  kernel_ = new TfLiteAsyncKernel();
  kernel_->kernel_data = this;
  kernel_->register_buffer = internal::RegisterBuffer;
  kernel_->register_buffer_slice = internal::RegisterBufferSlice;
  kernel_->unregister_buffer = internal::UnregisterBuffer;
  kernel_->supported_buffer_types = internal::SupportedBufferTypes;
  kernel_->supported_synchronizations = internal::SupportedSynchronizations;
  kernel_->reconcile_restrictions = internal::ReconcileRestrictions;
  kernel_->set_attributes = internal::SetAttributes;
  kernel_->prepare = internal::Prepare;
  kernel_->eval = internal::Eval;
  kernel_->wait = internal::Wait;
  kernel_->finish = internal::Finish;
}

}  // namespace delegates
}  // namespace tflite
