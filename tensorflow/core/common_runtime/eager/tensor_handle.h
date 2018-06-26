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
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
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

namespace tensorflow {

// Associates a Tensor and a Device, used in the eager runtime. Internal version
// of the TFE_TensorHandle struct and the python EagerTensor class
// (unrelated to python TensorHandle).
class TensorHandle : public core::RefCounted {
 public:
  TensorHandle(const Tensor& t, Device* d, Device* op_device, EagerContext* ctx)
      : dtype(t.dtype()),
        node_id(0),
        tensor_(t),
        device_(d),
        op_device_(op_device),
        remote_op_id_(-1),
        remote_output_num_(-1),
        ctx_(ctx),
        is_ready_(true) {}

  TensorHandle(uint64 node_id, DataType dtype, EagerContext* ctx)
      : dtype(dtype),
        node_id(node_id),
        tensor_(dtype),
        device_(nullptr),
        op_device_(nullptr),
        remote_op_id_(-1),
        remote_output_num_(-1),
        ctx_(ctx),
        is_ready_(ctx == nullptr) {
    DCHECK_GT(node_id, 0);
  }

  // Remote tensor handle constructor.
  TensorHandle(int64 op_id, int32 output_num, DataType dtype,
               std::function<void()> call_on_destroy, Device* d,
               Device* op_device, EagerContext* ctx)
      : dtype(dtype),
        node_id(0),
        device_(d),
        op_device_(op_device),
        remote_op_id_(op_id),
        remote_output_num_(output_num),
        call_on_destroy_(std::move(call_on_destroy)),
        ctx_(ctx),
        is_ready_(true) {
    DCHECK(IsRemote()) << "Op ID and output num should be >= 0. Op ID: "
                       << op_id << ", Output num: " << output_num;
  }

  ~TensorHandle() override {
    if (call_on_destroy_) {
      call_on_destroy_();
    }
  }

  Status Tensor(const tensorflow::Tensor** t);

  Status Device(tensorflow::Device** d);

  Status OpDevice(tensorflow::Device** d);

  Status TensorAndDevice(const tensorflow::Tensor** tensor,
                         tensorflow::Device** device,
                         tensorflow::Device** op_device);

  // Return the op_id and output num if the handle refers to a remote tensor.
  Status RemoteAddress(int64* op_id, int32* output_num);

  // Note that this can be called at most once, and only on non-ready handles,
  // and makes them ready.
  void SetTensorAndDevice(const tensorflow::Tensor& tensor,
                          tensorflow::Device* device,
                          tensorflow::Device* op_device);

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

 private:
  // If the contents of the Tensor pointed to by this handle is yet to be
  // computed by a EagerNode, this function will block till that compuatation is
  // done and the handle is "ready".
  Status WaitReady();

  bool IsReady();

  bool IsRemote();

  // Id for the EagerNode that will compute the value pointed to by this handle.
  // If the value is 0, the handle is already ready, but not vice-versa.
  const uint64 node_id;

  tensorflow::Tensor tensor_;

  // TODO(ashankar): device_ == nullptr iff local CPU
  // This was expedient, but perhaps worth revisiting ('device_' should always
  // be a valid pointer?)
  // This can be done if TFE_NewOp() and the TFE_TensorHandle constructors are
  // provided with the appropriate TFE_Context.
  //
  // TODO(ashankar): Reference count TFE_Context to ensure that 'device_' of a
  // TFE_TensorHandle does not outlive the TFE_Context from which it came?
  tensorflow::Device* device_;

  // Device in which the op producing this tensor was executed. Equals to
  // device_ for constant tensors.
  tensorflow::Device* op_device_;

  // IDs required when this class is representing a remote tensor handle.
  const int64 remote_op_id_;
  const int32 remote_output_num_;

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
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
