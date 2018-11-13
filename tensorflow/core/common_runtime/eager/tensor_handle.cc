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

#include "tensorflow/core/common_runtime/copy_tensor.h"
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

Status TensorHandle::Device(tensorflow::Device** d) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *d = device_;
  return Status::OK();
}

Status TensorHandle::OpDevice(tensorflow::Device** d) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *d = op_device_;
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
    CHECK(remote_shape_ != nullptr);
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

Status TensorHandle::RemoteAddress(int64* op_id, int32* output_num) {
  if (!IsRemote()) {
    return errors::FailedPrecondition(
        "This TensorHandle refers to a local tensor handle");
  }
  *op_id = remote_op_id_;
  *output_num = remote_output_num_;

  return Status::OK();
}

void TensorHandle::SetTensorAndDevice(const tensorflow::Tensor& tensor,
                                      tensorflow::Device* device,
                                      tensorflow::Device* op_device) {
  mutex_lock l(ctx_mutex_);
  DCHECK(node_id_ > 0 && !is_ready_)
      << "SetTensorAndDevice should be only called  "
      << "on non-ready handles.";
  is_ready_ = true;
  tensor_ = tensor;
  device_ = device;
  op_device_ = op_device;
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
  // both_on_cpu can be true and yet is_same_device is false, if one of src/dst
  // has device type XLA_CPU, and the other CPU.
  const bool both_on_cpu = src_cpu && dst_cpu;
  if (is_same_device || both_on_cpu) {
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

}  // namespace tensorflow
