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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
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
 public:
  TensorHandle(const Tensor& t, Device* d, Device* op_device,
               EagerContext* ctx);
  TensorHandle(uint64 node_id, Device* d, Device* op_device,
               Device* resource_device, DataType dtype, EagerContext* ctx);

  // Remote tensor handle constructor.
  TensorHandle(int64 op_id, int32 output_num, uint64 remote_shape_node_id,
               DataType dtype, std::function<void()> call_on_destroy, Device* d,
               Device* op_device, Device* resource_device, EagerContext* ctx);

  // Symbolic tensor constructor.
  TensorHandle(OutputGraphNode symbolic_tensor, DataType dtype);

  ~TensorHandle() override {
    VLOG(1) << "Deleting internal TensorHandle " << this;
    if (call_on_destroy_) {
      call_on_destroy_();
    }
  }

  Status Tensor(const tensorflow::Tensor** t);

  Status TensorValue(tensorflow::TensorValue* t);

  tensorflow::Device* device() const { return device_; }
  tensorflow::Device* op_device() const { return op_device_; }
  tensorflow::Device* resource_device() const { return resource_device_; }

  Status TensorAndDevice(const tensorflow::Tensor** tensor,
                         tensorflow::Device** device,
                         tensorflow::Device** op_device);

  Status Shape(tensorflow::TensorShape* shape);

  Status NumDims(int* num_dims);
  Status Dim(int dim_index, int64* dim);
  Status NumElements(int64* num_elements);

  // Return the op_id and output num if the handle refers to a remote tensor.
  Status RemoteAddress(int64* op_id, int32* output_num);

  // Note that this can be called at most once, and only on non-ready handles,
  // and makes them ready.
  void SetTensor(const tensorflow::Tensor& tensor);

  Status CopyToDevice(EagerContext* ctx, tensorflow::Device* dstd,
                      TensorHandle** output);

  // Warning: can return nullptr for CPU tensors.
  EagerContext* Context() {
    mutex_lock ml(ctx_mutex_);
    return ctx_;
  }

  // dtype for the handle. It must be the same as t.dtype() once the handle is
  // ready.
  const DataType dtype;

  void SetRemoteShape(std::unique_ptr<TensorShape> remote_shape) {
    remote_shape_ = std::move(remote_shape);
  }

  bool OnHostCPU() {
    mutex_lock ml(ctx_mutex_);
    return device_ == nullptr ||
           (ctx_ == nullptr || ctx_->HostCPU() == device_);
  }

  bool IsRemote();

  OutputGraphNode* getSymbolicTensor() const { return symbolic_tensor.get(); }

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
  Status WaitForNode(uint64 node_id, bool return_if_is_ready);

  bool IsReady();

  // Id for the EagerNode that will compute the value pointed to by this handle.
  // If the value is 0, the handle is already ready, but not vice-versa.
  const uint64 node_id_;

  tensorflow::Tensor tensor_;

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

  // IDs required when this class is representing a remote tensor handle.
  const int64 remote_op_id_;
  const int32 remote_output_num_;
  std::unique_ptr<TensorShape> remote_shape_;
  const uint64 remote_shape_node_id_;

  // A callback that is executed when the class is destroyed.
  //
  // This is currently used for remote tensor handles.
  const std::function<void()> call_on_destroy_;

  mutex ctx_mutex_;

  // `ctx` is only guaranteed to be set if the handle is not "ready". This is
  // typically true when the handle was produced during async execution.
  // `ctx` object is not owned and should outlive this handle.
  EagerContext* ctx_ GUARDED_BY(ctx_mutex_);
  bool is_ready_ GUARDED_BY(ctx_mutex_);

  // When non-NULL, this tensor handle instance represents a symbolic tensor
  // (corresponding to a graph node), whose concrete value is to be produced by
  // executing that graph node.
  std::unique_ptr<OutputGraphNode> symbolic_tensor;

  // If this TensorHandle is 1) a local tensor, and 2) a resource variable, we
  // will store data type and shape of the resource variable to
  // `resource_dtype_and_shape_`.
  std::pair<DataType, TensorShape> resource_dtype_and_shape_
      GUARDED_BY(ctx_mutex_);
  // `resource_dtype_and_shape_status_` stores whether we succeeded to get data
  // type of shape of this TensorHandle.
  Status resource_dtype_and_shape_status_ GUARDED_BY(ctx_mutex_);
  // `resource_dtype_and_shape_initialized_` indicates whether we have tried to
  // get `resource_dtype_and_shape_`.
  bool resource_dtype_and_shape_initialized_ GUARDED_BY(ctx_mutex_);
};

// If tensor's dtype is DT_RESOURCE, returns the device backing the resource.
// Else, returns nullptr.
Device* GetResourceDevice(const Tensor& t, EagerContext* ctx);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
