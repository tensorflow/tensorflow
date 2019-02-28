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
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
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

TensorHandle::TensorHandle(const class Tensor& t, Device* d, Device* op_device,
                           EagerContext* ctx)
    : dtype(t.dtype()),
      node_id_(0),
      tensor_(t),
      device_(d),
      op_device_(op_device),
      resource_device_(GetResourceDevice(t, ctx)),
      remote_op_id_(-1),
      remote_output_num_(-1),
      remote_shape_node_id_(-1),
      ctx_(ctx),
      is_ready_(true) {}

TensorHandle::TensorHandle(uint64 node_id, Device* d, Device* op_device,
                           Device* resource_device, DataType dtype,
                           EagerContext* ctx)
    : dtype(dtype),
      node_id_(node_id),
      tensor_(dtype),
      device_(d),
      op_device_(op_device),
      resource_device_(resource_device),
      remote_op_id_(-1),
      remote_output_num_(-1),
      remote_shape_node_id_(-1),
      ctx_(ctx),
      is_ready_(ctx == nullptr) {
  DCHECK_GT(node_id_, 0);
  DCHECK(dtype == DT_RESOURCE ? resource_device_ != nullptr
                              : resource_device_ == nullptr);
}

TensorHandle::TensorHandle(int64 op_id, int32 output_num,
                           uint64 remote_shape_node_id, DataType dtype,
                           std::function<void()> call_on_destroy, Device* d,
                           Device* op_device, Device* resource_device,
                           EagerContext* ctx)
    : dtype(dtype),
      node_id_(0),
      device_(d),
      op_device_(op_device),
      resource_device_(resource_device),
      remote_op_id_(op_id),
      remote_output_num_(output_num),
      remote_shape_node_id_(remote_shape_node_id),
      call_on_destroy_(std::move(call_on_destroy)),
      ctx_(ctx),
      is_ready_(true) {
  DCHECK(IsRemote()) << "Op ID and output num should be >= 0. Op ID: " << op_id
                     << ", Output num: " << output_num;
  DCHECK(dtype == DT_RESOURCE ? resource_device_ != nullptr
                              : resource_device_ == nullptr);
}

TensorHandle::TensorHandle(OutputGraphNode symbolic_tensor, DataType dtype)
    : dtype(dtype),
      node_id_(0),
      device_(nullptr),
      op_device_(nullptr),
      resource_device_(nullptr),
      remote_op_id_(-1),
      remote_output_num_(-1),
      remote_shape_node_id_(-1),
      ctx_(nullptr),
      is_ready_(true),
      symbolic_tensor(new OutputGraphNode(symbolic_tensor)) {}

bool TensorHandle::IsReady() {
  if (node_id_ == 0) return true;
  mutex_lock l(ctx_mutex_);
  return is_ready_;
}

bool TensorHandle::IsRemote() {
  return remote_op_id_ >= 0 && remote_output_num_ >= 0;
}

Status TensorHandle::WaitForNode(uint64 node_id, bool return_if_is_ready) {
  if (node_id == 0) return Status::OK();
  EagerExecutor* executor = nullptr;
  {
    mutex_lock l(ctx_mutex_);
    if (return_if_is_ready && is_ready_) return Status::OK();
    executor = ctx_->Executor();
  }
  return executor->WaitFor(node_id);
}

Status TensorHandle::WaitReady() { return WaitForNode(node_id_, true); }

Status TensorHandle::Tensor(const tensorflow::Tensor** t) {
  if (IsRemote()) {
    return errors::Unavailable(
        "Unable to get a tensor for a remote device. Please copy the tensor "
        "handle to a local device using TFE_TensorHandleCopyToDevice");
  }
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *t = &tensor_;
  return Status::OK();
}

Status TensorHandle::TensorValue(tensorflow::TensorValue* t) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *t = tensorflow::TensorValue(&tensor_);
  return Status::OK();
}

Status TensorHandle::TensorAndDevice(const tensorflow::Tensor** tensor,
                                     tensorflow::Device** device,
                                     tensorflow::Device** op_device) {
  if (IsRemote()) {
    return errors::Unavailable(
        "Unable to get a tensor for a remote device. Please copy the tensor "
        "handle to a local device using TFE_TensorHandleCopyToDevice");
  }
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *tensor = &tensor_;
  *device = device_;
  *op_device = op_device_;
  return Status::OK();
}

Status TensorHandle::Shape(tensorflow::TensorShape* shape) {
  if (IsRemote()) {
    TF_RETURN_IF_ERROR(WaitForNode(remote_shape_node_id_, false));
    CHECK(remote_shape_ != nullptr);
    *shape = *(remote_shape_.get());
  } else {
    TF_RETURN_IF_ERROR(WaitReady());
    DCHECK(IsReady());
    *shape = tensor_.shape();
  }
  return Status::OK();
}

Status TensorHandle::NumDims(int* num_dims) {
  if (IsRemote()) {
    TF_RETURN_IF_ERROR(WaitForNode(remote_shape_node_id_, false));
    *num_dims = remote_shape_->dims();
  } else {
    TF_RETURN_IF_ERROR(WaitReady());
    DCHECK(IsReady());
    DCHECK(num_dims != nullptr);

    *num_dims = tensor_.dims();
  }

  return Status::OK();
}

Status TensorHandle::Dim(int dim_index, int64* dim) {
  if (IsRemote()) {
    TF_RETURN_IF_ERROR(WaitForNode(remote_shape_node_id_, false));
    *dim = remote_shape_->dim_size(dim_index);
  } else {
    TF_RETURN_IF_ERROR(WaitReady());
    DCHECK(IsReady());
    DCHECK(dim != nullptr);

    *dim = tensor_.dim_size(dim_index);
  }

  return Status::OK();
}

Status TensorHandle::NumElements(int64* num_elements) {
  if (IsRemote()) {
    TF_RETURN_IF_ERROR(WaitForNode(remote_shape_node_id_, false));
    *num_elements = remote_shape_->num_elements();
  } else {
    TF_RETURN_IF_ERROR(WaitReady());
    DCHECK(IsReady());
    DCHECK(num_elements != nullptr);

    *num_elements = tensor_.NumElements();
  }

  return Status::OK();
}

Status TensorHandle::RemoteAddress(int64* op_id, int32* output_num) {
  if (!IsRemote()) {
    return errors::FailedPrecondition(
        "This TensorHandle refers to a local tensor handle");
  }
  *op_id = remote_op_id_;
  *output_num = remote_output_num_;

  return Status::OK();
}

void TensorHandle::SetTensor(const tensorflow::Tensor& tensor) {
  mutex_lock l(ctx_mutex_);
  DCHECK(node_id_ > 0 && !is_ready_) << "SetTensor should be only called  "
                                     << "on non-ready handles.";
  is_ready_ = true;
  tensor_ = tensor;
}

Status TensorHandle::CopyToDevice(EagerContext* ctx, tensorflow::Device* dstd,
                                  TensorHandle** output) {
  const tensorflow::Tensor* src = nullptr;
  tensorflow::Device* srcd = nullptr;
  // TODO(agarwal): src_opd is unused. Perhaps allow TensorAndDevice to accept
  // nullptr.
  tensorflow::Device* src_opd = nullptr;
  TF_RETURN_IF_ERROR(TensorAndDevice(&src, &srcd, &src_opd));
  if (srcd == nullptr) srcd = ctx->HostCPU();
  bool is_same_device = (srcd == dstd) || (srcd->name() == dstd->name());
  const bool dst_cpu = dstd->tensorflow_gpu_device_info() == nullptr;
  const bool src_cpu = srcd->tensorflow_gpu_device_info() == nullptr;
  if (is_same_device) {
    *output = new tensorflow::TensorHandle(*src, dstd, dstd, ctx);
    return tensorflow::Status::OK();
  }
  if (!dst_cpu && (src->dtype() != tensorflow::DT_VARIANT &&
                   !tensorflow::DataTypeCanUseMemcpy(src->dtype()))) {
    return tensorflow::errors::InvalidArgument(
        "Can't copy Tensor with type ",
        tensorflow::DataTypeString(src->dtype()), " to device ", dstd->name(),
        ".");
  }
  tensorflow::AllocatorAttributes attr;
  if (src->dtype() == tensorflow::DT_VARIANT) {
    attr.set_on_host(true);
  }
  tensorflow::Tensor dst(dstd->GetAllocator(attr), src->dtype(), src->shape());
  if (src->shape().num_elements() == 0) {
    dstd = dst_cpu ? nullptr : dstd;
    *output = new tensorflow::TensorHandle(dst, dstd, dstd, ctx);
    return tensorflow::Status::OK();
  }
  tensorflow::DeviceContext* src_device_context = nullptr;
  if (!src_cpu) {
    src_device_context = srcd->tensorflow_gpu_device_info()->default_context;
  }
  tensorflow::DeviceContext* dst_device_context = nullptr;
  if (!dst_cpu) {
    dst_device_context = dstd->tensorflow_gpu_device_info()->default_context;
  }
  // TODO(ashankar): The Sync() call below may be more aggressive than
  // necessary. It is based on knowledge of implementation details - that
  // GPU devices are implemented using 3 streams - one for host->device copies,
  // one for device->host copies and one for sending operations to the GPU.
  // With that setup, Sync()ing across all 3 streams should be sufficient
  // but more than necessary (since it waits for operations that might have
  // nothing to do with this tensor to complete).
  TF_RETURN_IF_ERROR(srcd->Sync());
  tensorflow::Notification n;
  tensorflow::Status status;
  tensorflow::CopyTensor::ViaDMA("copy", src_device_context, dst_device_context,
                                 srcd, dstd, tensorflow::AllocatorAttributes(),
                                 tensorflow::AllocatorAttributes(), src, &dst,
                                 0 /*dev_to_dev_stream_index*/,
                                 [&status, &n](const tensorflow::Status& s) {
                                   status = s;
                                   n.Notify();
                                 });
  n.WaitForNotification();
  if (status.ok()) {
    dstd = dst_cpu ? nullptr : dstd;
    *output = new tensorflow::TensorHandle(dst, dstd, dstd, ctx);
  }
  return status;
}

Device* GetResourceDevice(const Tensor& t, EagerContext* ctx) {
  if (t.dtype() != DT_RESOURCE) {
    return nullptr;
  }
  const ResourceHandle& resource_handle = t.flat<ResourceHandle>()(0);
  const auto& map = *ctx->device_map();
  auto it = map.find(resource_handle.device());
  DCHECK(it != map.end());
  return it->second;
}

string TensorHandle::DebugString() const {
  VLOG(1) << "Calling TensorHandle::DebugString() on " << this;

  if (symbolic_tensor) {
    return absl::Substitute("TF_Output($0, $1)", symbolic_tensor->oper,
                            symbolic_tensor->index);
  }

  string out;
  strings::StrAppend(&out, "Device: ", device_ ? device_->DebugString() : "[]");
  // Consider supporting non-CPU tensors (when device_ is non-NULL) if needed.
  strings::StrAppend(&out, ", Tensor: ", device_ ? "?" : tensor_.DebugString(),
                     "\n");
  return out;
}

}  // namespace tensorflow
