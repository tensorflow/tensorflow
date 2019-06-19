/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

struct TF_Operation;

namespace tensorflow {

// This struct is isomorphic to TF_Output, but we cannot use the latter here due
// to layering concerns (TF_Output is defined at the C API layer).
struct OutputGraphNode {
  TF_Operation* oper;
  int index;  // The index of the output within oper.
};

// Associates a Tensor and a Device, used in the eager runtime. Internal version
// of the TFE_TensorHandle struct and the python EagerTensor class
// (unrelated to python TensorHandle).
class TensorHandle : public core::RefCounted {
  // TensorHandle for dtype != DT_RESOURCE
  TensorHandle(std::unique_ptr<LocalTensorHandleData> t, DataType dtype,
               Device* d, Device* op_device, EagerContext* ctx);
  // TensorHandle for dtype == DT_RESOURCE
  TensorHandle(std::unique_ptr<LocalTensorHandleData> t,
               const ResourceHandle& resource_handle, Device* d,
               Device* op_device, EagerContext* ctx);
  TensorHandle(std::unique_ptr<AsyncLocalTensorHandleData> t, Device* d,
               Device* op_device, Device* resource_device, DataType dtype,
               EagerContext* ctx);

#if !defined(IS_MOBILE_PLATFORM)
  TensorHandle(std::unique_ptr<RemoteTensorHandleData> t, DataType dtype,
               Device* d, Device* resource_device, EagerContext* ctx);
  TensorHandle(std::unique_ptr<UnshapedRemoteTensorHandleData> t,
               DataType dtype, Device* d, Device* resource_device,
               EagerContext* ctx);
#endif  // IS_MOBILE_PLATFORM

 public:
  // TensorHandle with no assigned device
  static Status CreateLocalHandle(const class Tensor& t, TensorHandle** h);
  // TensorHandle with device == op_device
  static Status CreateLocalHandle(const class Tensor& t, Device* d,
                                  EagerContext* ctx, TensorHandle** h);
  static Status CreateLocalHandle(const class Tensor& t, Device* d,
                                  Device* op_device, EagerContext* ctx,
                                  TensorHandle** h);
  static Status CreateAsyncLocalHandle(Device* d, Device* op_device,
                                       Device* resource_device, DataType dtype,
                                       EagerContext* ctx, TensorHandle** h);
#if !defined(IS_MOBILE_PLATFORM)
  static Status CreateRemoteHandle(int64 op_id, int output_num,
                                   const TensorShape& shape,
                                   eager::EagerClient* eager_client,
                                   uint64 context_id, DataType dtype, Device* d,
                                   Device* resource_device, EagerContext* ctx,
                                   TensorHandle** h);
  static Status CreateRemoteHandle(std::unique_ptr<RemoteTensorHandleData> t,
                                   DataType dtype, Device* d,
                                   Device* resource_device, EagerContext* ctx,
                                   TensorHandle** h);
  static Status CreateUnshapedRemoteHandle(int64 op_id, int32 output_num,
                                           eager::EagerClient* eager_client,
                                           uint64 context_id, DataType dtype,
                                           Device* d, Device* resource_device,
                                           EagerContext* ctx, TensorHandle** h);
#endif  // IS_MOBILE_PLATFORM

  // Symbolic tensor constructor.
  TensorHandle(OutputGraphNode symbolic_tensor, DataType dtype);

  ~TensorHandle() override { VLOG(3) << "Deleting TensorHandle " << this; }

  Status Tensor(const tensorflow::Tensor** t);

  Status TensorValue(tensorflow::TensorValue* t);

  tensorflow::Device* device() const { return device_; }
  tensorflow::Device* op_device() const { return op_device_; }
  tensorflow::Device* resource_device() const { return resource_device_; }

  Status Shape(tensorflow::TensorShape* shape);

  Status NumDims(int* num_dims);
  Status Dim(int dim_index, int64* dim);
  Status NumElements(int64* num_elements);

#if !defined(IS_MOBILE_PLATFORM)
  bool HasRemoteMirror(Device* d);
  // TODO(gjn): Add Unshaped remote mirrors once EagerRemoteSendTensor supports
  // async execution and EagerRemoteExecute is mirror-aware.
  Status AddRemoteMirror(std::unique_ptr<RemoteTensorHandleData> t, Device* d);

  // Return the op_id and output num if the handle refers to a remote tensor.
  Status RemoteAddress(Device* d, int64* op_id, int32* output_num) const;

  // Called on an async remote tensor once it's shape has been determined. This
  // transitions the tensor handle from a non-ready to a ready state by
  // replacing the backing data abstraction to allow for the shape to be
  // queried.
  // This method or Poison must be called exactly once for remote tensors that
  // were created without a known shape.
  Status SetRemoteShape(const TensorShape& shape);
#endif

  // Sets the `tensor` for this async non-ready handle making it ready.
  // This method or Poison must be called exactly once for non-ready async
  // handles to make them ready.
  void SetTensor(const tensorflow::Tensor& tensor);

  // Poisons this non-ready handle with an error `status`.
  // Poisoning means that the handle will become ready and methods trying
  // to access the actual tensor or shape will return this error `status`.
  // Exactly one of SetTensor, SetRemoteShape, or Poison methods must be called
  // on a non-ready tensor.
  void Poison(Status status);

  Status CopyToDevice(EagerContext* ctx, tensorflow::Device* dstd,
                      TensorHandle** output);

  // Warning: can return nullptr for CPU tensors.
  EagerContext* Context() { return ctx_; }

  // dtype for the handle. It must be the same as t.dtype() once the handle is
  // ready.
  const DataType dtype;

  bool OnHostCPU() const {
    return device_ == nullptr ||
           (ctx_ != nullptr && ctx_->HostCPU() == device_);
  }

  bool IsRemote() const { return is_remote_; }

  OutputGraphNode* getSymbolicTensor() const { return symbolic_tensor_.get(); }

  string DebugString() const;

  // If this TensorHandle is 1) a local tensor, and 2) a resource variable,
  // return data type and shape of the resource variable.
  Status GetResourceVariableDtypeAndShape(
      std::pair<DataType, TensorShape>* result);

 private:
  // If the contents of the Tensor pointed to by this handle is yet to be
  // computed by a EagerNode, this function will block till that computation is
  // done and the handle is "ready".
  Status WaitReady();

  // TODO(ashankar): device_ == nullptr iff local CPU
  // This was expedient, but perhaps worth revisiting ('device_' should always
  // be a valid pointer?)
  // This can be done if TFE_NewOp() and the TFE_TensorHandle constructors are
  // provided with the appropriate TFE_Context.
  //
  // TODO(ashankar): Reference count TFE_Context to ensure that 'device_' of a
  // TFE_TensorHandle does not outlive the TFE_Context from which it came?
  tensorflow::Device* const device_;

  // Device in which the op producing this tensor was executed. Equals to
  // device_ for constant tensors.
  // Can be nullptr if the op producing this tensor was a function executed
  // with function library runtime or if this tensor represents a symbolic
  // tensor.
  tensorflow::Device* const op_device_;

  // If the tensor dtype is DT_RESOURCE, resource_device_ holds the device
  // backing the resource. Else resource_device_ is nullptr.
  tensorflow::Device* const resource_device_;

#if !defined(IS_MOBILE_PLATFORM)
  mutable mutex remote_mirrors_mutex_;
  std::map<tensorflow::Device*, std::unique_ptr<RemoteTensorHandleData>>
      remote_mirrors_ GUARDED_BY(remote_mirrors_mutex_);

  // IDs required when this class is representing a remote tensor handle.
  const int64 remote_op_id_;
  const int32 remote_output_num_;
  eager::EagerClient* remote_eager_client_;
  uint64 remote_context_id_;
#endif

  // `ctx` is only guaranteed to be set if the handle is not "ready". This is
  // typically true when the handle was produced during async execution.
  // `ctx` object is not owned and should outlive this handle.
  EagerContext* const ctx_;

  // Explanation for NOLINT below: absl has clang-tidy macro to rename
  // 'tensorflow::Notification' to 'absl::Notification'. TF does not use
  // absl::Notification in open source now, so we can't follow clang-tidy
  tensorflow::Notification is_ready_notification_;  // NOLINT
  // Does not need synchronization because it can be accessed only after
  // WaitReady() has returned. At that point, is_poisoned_ is immutable.
  Status is_poisoned_;
  const bool is_remote_;

  // When non-NULL, this tensor handle instance represents a symbolic tensor
  // (corresponding to a graph node), whose concrete value is to be produced by
  // executing that graph node.
  std::unique_ptr<OutputGraphNode> symbolic_tensor_;

  // If this TensorHandle is 1) a local tensor, and 2) a resource handle, we
  // we store the container and name to be able to get the data type and shape
  // in a call to GetResourceVariableDtypeAndShape.
  string resource_handle_container_;
  string resource_handle_name_;

  // The TensorHandleData can either represent a local or remote tensor handle.
  // Further, it can be in a non-ready state. It would become ready with a call
  // to either SetTensor or SetRemoteShape which replaces the underlying data
  // with a ready version of the tensor handle data.
  // Does not need synchronization because it can be accessed only after
  // WaitReady() has returned. At that point, tensor_handle_data_ is immutable.
  std::unique_ptr<TensorHandleData> tensor_handle_data_;
};

// Returns the device backing the resource. Else, returns nullptr.
Device* GetResourceDevice(const ResourceHandle& handle, EagerContext* ctx);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
