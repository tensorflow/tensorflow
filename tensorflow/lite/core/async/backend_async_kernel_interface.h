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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_BACKEND_ASYNC_KERNEL_INTERFACE_H_
#define TENSORFLOW_LITE_CORE_ASYNC_BACKEND_ASYNC_KERNEL_INTERFACE_H_

#include <vector>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
// TODO(b/191883048): This interface should only depend on C API instead of
// internal definitions.
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/common.h"

namespace tflite {
namespace delegates {

// A C++ wrapper around TfLiteAsyncKernel C API that delegate developers
// can use to add support for asynchronous execution.
// The implementation of `BackendAsyncKernelInterface` must be thread safe.
class BackendAsyncKernelInterface {
 public:
  BackendAsyncKernelInterface();
  virtual ~BackendAsyncKernelInterface() {
    if (kernel_) delete kernel_;
  }

  // Returns the TfLiteAsyncKernel instance.
  // kernel_ will be filled with the implementation of the class.
  virtual TfLiteAsyncKernel* kernel() { return kernel_; }

  // The following methods should be implemented to support buffer interop
  // and asynchronous execution.

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
  // In `attrs`, the application must provide the type of the buffer.
  // If additional attributes (e.g. padding, size) are provided, the backend
  // is responsible for validating those attributes to be compatible.
  // The backend will not own the actual buffer wrapped in `buffer`, but the
  // backend can choose to increase the ref count if underlying implementaion
  // supports that.
  virtual TfLiteStatus RegisterBuffer(TfLiteOpaqueContext* context,
                                      TfLiteIoType io_type,
                                      const TfLiteBackendBuffer* buffer,
                                      const TfLiteAttributeMap* attrs,
                                      TfLiteBufferHandle handle) = 0;

  // Registers a buffer slice from a previously registered memory.
  // `buffer` is the handle of the buffer pool previously registered.
  // `attrs` contains the information of the buffer slice.
  // `handle` is the buffer handle assigned by TfLite runtime to recognize
  // this piece of buffer.
  // NOTE: The backend is responsible to validate the slicing is "valid":
  // * The slicing is not nested from another slice. (i.e. the `buffer_pool` is
  //   a handle returned by `RegisterBuffer`.)
  // * The attributes of the slice (e.g. size, offset) is of valid values
  //   from the buffer pool.
  // If the `handle` is not recognized, returns error.
  virtual TfLiteStatus RegisterBufferSlice(TfLiteOpaqueContext* context,
                                           TfLiteBufferHandle buffer_pool,
                                           const TfLiteAttributeMap* attrs,
                                           TfLiteBufferHandle handle) = 0;

  // Unregisters a buffer or a buffer slice.
  // `handle` is a buffer handle previously assigned via register_* calls.
  // If the `handle` is not recognized, returns error.
  // Unregistering the buffer does not mean deallocating the buffer. However
  // the backend need to reduce the ref-count if ref counting is performed
  // during `Register*` calls.
  virtual TfLiteStatus UnregisterBuffer(TfLiteOpaqueContext* context,
                                        TfLiteBufferHandle handle) = 0;

  // Reconciliations
  // ===================
  // Inspects the buffer types supported by the backend.
  // `io_type` specify whether the call returns supported input or output
  // buffer.
  virtual std::vector<const char*> SupportedBufferTypes(
      TfLiteIoType io_type) const = 0;

  // Inspects the sync object types supported by the backend.
  // `io_type` specify whether the call returns supported input or output
  // sync object.
  virtual std::vector<const char*> SupportedSynchronizations(
      TfLiteIoType io_type) const = 0;

  // Reconciles buffer or sync attributes for tensor at tensor_index.
  // Fills `merged` with reconciled attributes.
  // If `conflict` is provided, conflicting attributes will be provided there.
  // If the type of the `user_provided_attributes` is not recognizable, returns
  // error.
  // If any of the attribute in the `user_provided_attributes` is not
  // recognizable skip this attribute.
  // Returns true if the attribute map type is recognizable and there's no
  // conflicting attribute.
  virtual bool ReconcileRestrictions(
      TfLiteOpaqueContext* context, TfLiteOpaqueNode* node, int tensor_index,
      const TfLiteAttributeMap* user_provided_attributes,
      TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const = 0;

  // Sets the input / output buffer / sync attributes.
  // Backend kernel will check the input attributes covers all the requirements.
  // A typical workflow is for callers call Reconcile*Restrictions method
  // above to have a merged attribute list, check all restrictions are met
  // and set input / output attribute here.
  // Returns TfLiteOk if provided `attrs` covers all requirements.
  virtual TfLiteStatus SetAttributes(TfLiteOpaqueContext* context,
                                     TfLiteOpaqueNode* node, int tensor_index,
                                     const TfLiteAttributeMap* attrs) = 0;

  // Prepares the kernel using the information from Set[In|Out]putAttributes
  // call above.
  virtual TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                               TfLiteOpaqueNode* node) = 0;

  // Execution methods
  // =============================

  // Schedules an execution with the information provided in task.
  // The application is responsible for filling out buffer and sync mappings
  // to tensors.
  // Backend will set the sync ptr for related tensors if requested.
  // i.e. SetOutputAttributes has sync implementation requested, and
  // the TfLiteSynchronization is not null for the tensor in `task`.
  //
  // TfLite runtime guarantees that the task is in ready state (i.e. no
  // un-ended execution for this task).
  //
  // Input synchronizations:
  // If the synchronization of a input tensor is `kTfLiteSyncTypeNoSyncObj`
  // type or it's nullptr, it means the data is ready during Eval call.
  // If not, data will be available when the synchronization signals and the
  // backend is responsible for closing the underlying synchronization.
  // The backend is responsible for dedupping the input sync.
  //
  // Output synchronizations:
  // If the synchronization type is `kTfLiteSyncTypeNoSyncObj` or is nullptr,
  // the backend does not need to provide synchronization objects to the user.
  // Otherwise, the backend need to provide the sync according to the sync type
  // provided. The underlying sync object will be closed by the app (or
  // downstream components).
  // If there are multiple non-nullptr kTfLiteSynchronization provided for
  // different output tensors, the backend is responsible for duplicating the
  // synchronization.
  // TODO(b/191883048): What if the sync fence is not dup-able?
  //
  // Returns kTfLiteOk if the execution is successfully scheduled.
  virtual TfLiteStatus Eval(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node,
                            TfLiteExecutionTask* task) = 0;

  // Waits on the execution scheduled using the task to finish.
  // TfLite runtime guarantees that the task has an un-ended execution.
  // Returns kTfLiteOk if the task is finished (w/ or w/o blocking).
  virtual TfLiteStatus Wait(TfLiteOpaqueContext* context,
                            TfLiteExecutionTask* task) = 0;

  // Finishes the task and clean up allocated resources for the task.
  // May block if there's pending executions.
  // Returns kTfLiteOk if there's no error.
  virtual TfLiteStatus Finish(TfLiteOpaqueContext* context,
                              TfLiteExecutionTask* task) = 0;

 protected:
  TfLiteAsyncKernel* kernel_ = nullptr;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_BACKEND_ASYNC_KERNEL_INTERFACE_H_
