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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_ASYNC_KERNEL_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_ASYNC_ASYNC_KERNEL_INTERNAL_H_

#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

typedef struct TfLiteAttributeMap TfLiteAttributeMap;
typedef struct TfLiteBackendBuffer TfLiteBackendBuffer;
typedef struct TfLiteExecutionTask TfLiteExecutionTask;

struct TfLiteAsyncKernel {
  // Stores the arbitrary data used to identify the async kernel it self.
  // Filled by the backend delegate.
  void* kernel_data;

  // Buffer operations
  // ======================
  // Registers the buffer to `handle`.
  // `buffer` and `attrs` lifespan is not gauranteed after the function call.
  // kernels should read the stored attributes instead of caching the
  // attribute map.
  // `io_type` specifies whether this buffer is used as an input buffer
  // or an output buffer. If a buffer is both used as input and output,
  // specify it as output. Not null.
  // `attrs` describes the attributes of the buffer. It's gauranteed to be
  // of kTfLiteBufferAttrMap type and not null.
  // `handle` is the buffer handle assigned by TfLite runtime to recognize
  // this piece of buffer.
  TfLiteStatus (*register_buffer)(TfLiteAsyncKernel* async_kernel,
                                  TfLiteOpaqueContext* context, int32_t io_type,
                                  const TfLiteBackendBuffer* buffer,
                                  const TfLiteAttributeMap* attrs,
                                  TfLiteBufferHandle handle) = nullptr;

  // Registers a buffer slice from a previously registered memory.
  // `buffer` is the handle of the buffer pool previously registered.
  // `attrs` contains the information of the buffer slice.
  // `handle` is the buffer handle assigned by TfLite runtime to recognize
  // this piece of buffer.
  // If the `handle` is not recognized, returns error.
  TfLiteStatus (*register_buffer_slice)(TfLiteAsyncKernel* async_kernel,
                                        TfLiteOpaqueContext* context,
                                        TfLiteBufferHandle buffer_pool,
                                        const TfLiteAttributeMap* attrs,
                                        TfLiteBufferHandle handle) = nullptr;

  // Unregisters a buffer or a buffer slice.
  // `handle` is a buffer handle previously assigned via register_* calls.
  // If the `handle` is not recognized, returns error.
  TfLiteStatus (*unregister_buffer)(TfLiteAsyncKernel* async_kernel,
                                    TfLiteOpaqueContext* context,
                                    TfLiteBufferHandle handle) = nullptr;

  // Reconciliations
  // ===================
  // Inspects the buffer types supported by the backend.
  // `io_type` specify whether the call returns supported input or output
  // buffer.
  std::vector<const char*> (*supported_buffer_types)(
      const TfLiteAsyncKernel* async_kernel, int32_t io_type) = nullptr;

  // Inspects the sync object types supported by the backend.
  // `io_type` specify whether the call returns supported input or output
  // sync object.
  std::vector<const char*> (*supported_synchronizations)(
      const TfLiteAsyncKernel* async_kernel, int32_t io_type) = nullptr;

  // Reconciles buffer or sync attributes for tensor at tensor_index.
  // Fills `merged` with reconciled attributes.
  // If `conflict` is provided, conflicting attributes will be provided there.
  // Returns true if there's no conflict.
  bool (*reconcile_restrictions)(
      const TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
      TfLiteOpaqueNode* node, int tensor_index,
      const TfLiteAttributeMap* user_provided_attributes,
      TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) = nullptr;

  // Sets the input / output buffer / sync attributes.
  // Backend kernel will check the input attributes covers all the requirements.
  // A typical workflow is for callers call Reconcile*Restrictions method
  // above to have a merged attribute list, check all restrictions are met
  // and set input / output attribute here.
  // Returns TfLiteOk if provided `attrs` covers all requirements.
  TfLiteStatus (*set_attributes)(TfLiteAsyncKernel* async_kernel,
                                 TfLiteOpaqueContext* context,
                                 TfLiteOpaqueNode* node, int tensor_index,
                                 const TfLiteAttributeMap* attrs) = nullptr;

  // Prepares the kernel using the information from Set[In|Out]putAttributes
  // call above.
  TfLiteStatus (*prepare)(TfLiteAsyncKernel* async_kernel,
                          TfLiteOpaqueContext* context,
                          TfLiteOpaqueNode* node) = nullptr;
  // Execution methods
  // =============================

  // Schedules an execution with the information provided in task.
  // The application is responsible for filling out buffer and sync mappings
  // to tensors.
  // Backend will set the sync ptr for related tensors if requested.
  // i.e. SetOutputAttributes has sync implementation requested, and
  // the TfLiteSynchronization is not null for the tensor in `task`.
  // Returns kTfLiteOk if the execution is successfully scheduled.
  TfLiteStatus (*eval)(TfLiteAsyncKernel* async_kernel,
                       TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                       TfLiteExecutionTask* task) = nullptr;

  // Waits on the execution scheduled using the task to finish.
  // Returns kTfLiteOk if the task is finished (w/ or w/o blocking).
  TfLiteStatus (*wait)(TfLiteAsyncKernel* async_kernel,
                       TfLiteOpaqueContext* context,
                       TfLiteExecutionTask* task) = nullptr;

  // Finishes the task and clean up allocated resources for the task.
  // May block if there's pending executions.
  // Returns kTfLiteOk if there's no error.
  TfLiteStatus (*finish)(TfLiteAsyncKernel* async_kernel,
                         TfLiteOpaqueContext* context,
                         TfLiteExecutionTask* task) = nullptr;
};

#endif  // TENSORFLOW_LITE_CORE_ASYNC_ASYNC_KERNEL_INTERNAL_H_
