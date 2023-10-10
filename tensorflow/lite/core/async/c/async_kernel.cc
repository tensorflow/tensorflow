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
#include "tensorflow/lite/core/async/c/async_kernel.h"

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/c/types.h"

TfLiteAsyncKernel* TfLiteAsyncKernelCreate(void* kernel_data) {
  TfLiteAsyncKernel* ret = new TfLiteAsyncKernel{};
  if (!ret) return nullptr;
  ret->kernel_data = kernel_data;
  return ret;
}

void* TfLiteAsyncKernelGetKernelData(const TfLiteAsyncKernel* async_kernel) {
  if (!async_kernel) return nullptr;
  return async_kernel->kernel_data;
}

void TfLiteAsyncKernelSetRegisterBuffer(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*register_buffer)(
        TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
        TfLiteIoType io_type, const TfLiteBackendBuffer* buffer,
        const TfLiteAttributeMap* attrs, TfLiteBufferHandle handle)) {
  if (!async_kernel) return;
  async_kernel->register_buffer = register_buffer;
}

void TfLiteAsyncKernelSetRegisterBufferSlice(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*register_buffer_slice)(TfLiteAsyncKernel* async_kernel,
                                          TfLiteOpaqueContext* context,
                                          TfLiteBufferHandle buffer_pool,
                                          const TfLiteAttributeMap* attrs,
                                          TfLiteBufferHandle handle)) {
  if (!async_kernel) return;
  async_kernel->register_buffer_slice = register_buffer_slice;
}

void TfLiteAsyncKernelSetUnregisterBuffer(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*unregister_buffer)(TfLiteAsyncKernel* async_kernel,
                                      TfLiteOpaqueContext* context,
                                      TfLiteBufferHandle handle)) {
  if (!async_kernel) return;
  async_kernel->unregister_buffer = unregister_buffer;
}

void TfLiteAsyncKernelSetSupportedBufferTypes(
    TfLiteAsyncKernel* async_kernel,
    void (*supported_buffer_types)(const TfLiteAsyncKernel* async_kernel,
                                   TfLiteIoType io_type,
                                   const char* const** types,
                                   size_t* n_types)) {
  if (!async_kernel) return;
  async_kernel->supported_buffer_types = supported_buffer_types;
}

void TfLiteAsyncKernelSetSupportedSynchronizations(
    TfLiteAsyncKernel* async_kernel,
    void (*supported_synchronizations)(const TfLiteAsyncKernel* async_kernel,
                                       TfLiteIoType io_type,
                                       const char* const** types,
                                       size_t* n_types)) {
  if (!async_kernel) return;
  async_kernel->supported_synchronizations = supported_synchronizations;
}

void TfLiteAsyncKernelSetReconcileRestrictions(
    TfLiteAsyncKernel* async_kernel,
    bool (*reconcile_restrictions)(
        const TfLiteAsyncKernel* async_kernel,
        const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node,
        int tensor_index, const TfLiteAttributeMap* user_provided_attributes,
        TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict)) {
  if (!async_kernel) return;
  async_kernel->reconcile_restrictions = reconcile_restrictions;
}

void TfLiteAsyncKernelSetSetAttributes(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*set_attributes)(TfLiteAsyncKernel* async_kernel,
                                   TfLiteOpaqueContext* context,
                                   TfLiteOpaqueNode* node, int tensor_index,
                                   const TfLiteAttributeMap* attrs)) {
  if (!async_kernel) return;
  async_kernel->set_attributes = set_attributes;
}

void TfLiteAsyncKernelSetPrepare(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*prepare)(TfLiteAsyncKernel* async_kernel,
                            TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node)) {
  if (!async_kernel) return;
  async_kernel->prepare = prepare;
}

void TfLiteAsyncKernelSetEval(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*eval)(TfLiteAsyncKernel* async_kernel,
                         TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                         TfLiteExecutionTask* task)) {
  if (!async_kernel) return;
  async_kernel->eval = eval;
}

void TfLiteAsyncKernelSetWait(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*wait)(TfLiteAsyncKernel* async_kernel,
                         TfLiteOpaqueContext* context,
                         TfLiteExecutionTask* task)) {
  if (!async_kernel) return;
  async_kernel->wait = wait;
}

void TfLiteAsyncKernelSetFinish(
    TfLiteAsyncKernel* async_kernel,
    TfLiteStatus (*finish)(TfLiteAsyncKernel* async_kernel,
                           TfLiteOpaqueContext* context,
                           TfLiteExecutionTask* task)) {
  if (!async_kernel) return;
  async_kernel->finish = finish;
}

void TfLiteAsyncKernelDelete(TfLiteAsyncKernel* async_kernel) {
  if (!async_kernel) return;
  delete async_kernel;
}
