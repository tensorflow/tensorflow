/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_INTERNAL_H_

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/runtime.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"


struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy policy{
      TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32};
};

struct TFE_Context {
  explicit TFE_Context(const tensorflow::SessionOptions& opts,
                       TFE_ContextDevicePlacementPolicy default_policy,
                       bool async,
                       std::unique_ptr<tensorflow::DeviceMgr> device_mgr,
                       tensorflow::Rendezvous* rendezvous)
      : context(opts,
                static_cast<tensorflow::ContextDevicePlacementPolicy>(
                    default_policy),
                async, std::move(device_mgr), rendezvous) {}

  tensorflow::EagerContext context;
};

struct TFE_TensorHandle : public tensorflow::core::RefCounted {
 public:
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d,
                   tensorflow::Device* op_device)
      : dtype(t.dtype()),
        node_id(0),
        tensor_(t),
        device_(d),
        op_device_(op_device),
        ctx_(nullptr) {}

  TFE_TensorHandle(tensorflow::uint64 node_id, tensorflow::DataType dtype,
                   TFE_Context* ctx)
      : dtype(dtype),
        node_id(node_id),
        tensor_(dtype),
        device_(nullptr),
        op_device_(nullptr),
        ctx_(ctx) {
    DCHECK_GT(node_id, 0);
  }

  ~TFE_TensorHandle() override {}

  tensorflow::Status Tensor(const tensorflow::Tensor** t);

  tensorflow::Status Device(tensorflow::Device** d);

  tensorflow::Status OpDevice(tensorflow::Device** d);

  tensorflow::Status TensorAndDevice(const tensorflow::Tensor** tensor,
                                     tensorflow::Device** device,
                                     tensorflow::Device** op_device);

  // Note that this can be called at most once, and only on non-ready handles,
  // and makes them ready.
  void SetTensorAndDevice(const tensorflow::Tensor& tensor,
                          tensorflow::Device* device,
                          tensorflow::Device* op_device);

  // dtype for the handle. It must be the same as t.dtype() once the handle is
  // ready.
  const tensorflow::DataType dtype;

 private:
  // If the contents of the Tensor pointed to by this handle is yet to be
  // computed by a EagerNode, this function will block till that compuatation is
  // done and the handle is "ready".
  tensorflow::Status WaitReady();

  bool IsReady();

  // Id for the EagerNode that will compute the value pointed to by this handle.
  // If the value is 0, the handle is already ready, but not vice-versa.
  const tensorflow::uint64 node_id;

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

  tensorflow::mutex ctx_mutex_;

  // `ctx` is only guaranteed to be set if the handle is not "ready". This is
  // typically true when the handle was produced during async execution.
  // `ctx` object is not owned and should outlive this handle.
  TFE_Context* ctx_ GUARDED_BY(ctx_mutex_);
};

struct TFE_Op {
  // t is NULL iff the TFE_Op corresponds to a TensorFlow function instead of a
  // primitive operation.
  TFE_Op(TFE_Context* ctx, const char* op, const tensorflow::AttrTypeMap* t)
      : ctx(ctx), name(op), attrs(op), attr_types(t), device(nullptr) {}

  ~TFE_Op();

  bool const is_function() const { return attr_types == nullptr; }

  TFE_Context* ctx;  // Must outlive the TFE_Op.
  const tensorflow::string name;
  tensorflow::AttrBuilder attrs;
  const tensorflow::AttrTypeMap* attr_types;
  tensorflow::gtl::InlinedVector<TFE_TensorHandle*, 4> inputs;
  tensorflow::Device* device;
  bool use_xla = false;
};

namespace tensorflow {
// Set an AttrValue on the op. Doesn't handle the list types.
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status);
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
