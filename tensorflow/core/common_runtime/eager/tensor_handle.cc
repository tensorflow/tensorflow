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
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
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

Status TensorHandle::GetResourceVariableDtypeAndShape(
    std::pair<DataType, TensorShape>* result) {
  if (IsRemote()) {
    return errors::Unimplemented(
        "Getting resource data type and shape for a remote tensor is not "
        "implemented yet");
  }

  if (dtype != DT_RESOURCE) {
    return errors::InvalidArgument(
        "TensorHandle::GetResourceDtypeAndShape should be called on tensor "
        "handles with data type DT_RESOURCE. Actual tensor: ",
        dtype);
  }

  // Wait for this TensorHandle to be ready.
  TF_RETURN_IF_ERROR(WaitReady());

  // Try to get resource data type and shape from ResourceMgr.
  if (!resource_device_) {
    return errors::Internal("Cannot get resource device for tensor ",
                            tensor_handle_data_->DebugString());
  }
  ResourceMgr* resource_mgr = resource_device_->resource_manager();
  if (!resource_mgr) {
    return errors::Internal("Cannot get ResourceMgr for device ",
                            resource_device_->DebugString());
  }
  Var* resource_var;
  // Here we do not differentiate between "resource does not exist" and
  // "resource is not a resource variable".
  // ResourceMgr uses resource's C++ class name as part of lookup key, so we
  // must provide the correct C++ class when calling Lookup().
  Status s = resource_mgr->Lookup(resource_handle_container_,
                                  resource_handle_name_, &resource_var);
  if (!s.ok()) {
    return errors::InvalidArgument(
        "ResourceHandle does not exist, or is not a resource variable: ",
        resource_handle_container_, resource_handle_name_);
  }
  core::ScopedUnref unref(resource_var);

  // Return the result.
  *result = std::make_pair(resource_var->tensor()->dtype(),
                           resource_var->tensor()->shape());
  return Status::OK();
}

TensorHandle::TensorHandle(const class Tensor& t, Device* d, Device* op_device,
                           EagerContext* ctx)
    : dtype(t.dtype()),
      tensor_handle_data_(absl::make_unique<LocalTensorHandleData>(t)),
      device_(d),
      op_device_(op_device),
      resource_device_(GetResourceDevice(t, ctx)),
#if !defined(IS_MOBILE_PLATFORM)
      remote_op_id_(-1),
      remote_output_num_(-1),
#endif
      ctx_(ctx),
      is_ready_(true),
      is_remote_(false) {
  if (dtype == DT_RESOURCE) {
    const ResourceHandle& resource_handle = t.flat<ResourceHandle>()(0);
    resource_handle_container_ = resource_handle.container();
    resource_handle_name_ = resource_handle.name();
  }
}

TensorHandle::TensorHandle(uint64 node_id, Device* d, Device* op_device,
                           Device* resource_device, DataType dtype,
                           EagerContext* ctx)
    : dtype(dtype),
      tensor_handle_data_(
          absl::make_unique<AsyncLocalTensorHandleData>(node_id, ctx)),
      device_(d),
      op_device_(op_device),
      resource_device_(resource_device),
#if !defined(IS_MOBILE_PLATFORM)
      remote_op_id_(-1),
      remote_output_num_(-1),
#endif
      ctx_(ctx),
      is_ready_(false),
      is_remote_(false) {
  DCHECK(dtype == DT_RESOURCE ? resource_device_ != nullptr
                              : resource_device_ == nullptr);
}

#if !defined(IS_MOBILE_PLATFORM)
TensorHandle::TensorHandle(int64 op_id, int output_num,
                           const TensorShape& shape,
                           eager::EagerClient* eager_client, uint64 context_id,
                           DataType dtype, Device* d, Device* resource_device,
                           EagerContext* ctx)
    : dtype(dtype),
      tensor_handle_data_(absl::make_unique<RemoteTensorHandleData>(
          op_id, output_num, shape, eager_client, context_id, ctx)),
      device_(d),
      op_device_(d),
      resource_device_(resource_device),
      remote_op_id_(op_id),
      remote_output_num_(output_num),
      remote_eager_client_(eager_client),
      remote_context_id_(context_id),
      ctx_(ctx),
      is_ready_(true),
      is_remote_(true) {}

TensorHandle::TensorHandle(int64 op_id, int32 output_num, uint64 shape_node_id,
                           eager::EagerClient* eager_client, uint64 context_id,
                           DataType dtype, Device* d, Device* resource_device,
                           EagerContext* ctx)
    : dtype(dtype),
      tensor_handle_data_(absl::make_unique<UnshapedRemoteTensorHandleData>(
          op_id, output_num, shape_node_id, eager_client, context_id, ctx)),
      device_(d),
      op_device_(d),
      resource_device_(resource_device),
      remote_op_id_(op_id),
      remote_output_num_(output_num),
      remote_eager_client_(eager_client),
      remote_context_id_(context_id),
      ctx_(ctx),
      is_ready_(false),
      is_remote_(true) {
  DCHECK(dtype == DT_RESOURCE ? resource_device_ != nullptr
                              : resource_device_ == nullptr);
}
#endif

TensorHandle::TensorHandle(OutputGraphNode symbolic_tensor, DataType dtype)
    : dtype(dtype),
      device_(nullptr),
      op_device_(nullptr),
      resource_device_(nullptr),
#if !defined(IS_MOBILE_PLATFORM)
      remote_op_id_(-1),
      remote_output_num_(-1),
#endif
      ctx_(nullptr),
      is_ready_(true),
      is_remote_(false),
      symbolic_tensor_(new OutputGraphNode(symbolic_tensor)) {
}

Status TensorHandle::WaitReady() {
  while (true) {
    {
      tf_shared_lock l(ready_mutex_);
      if (is_ready_) return Status::OK();
    }
    TF_RETURN_IF_ERROR(tensor_handle_data_->WaitReady());
  }

  return Status::OK();
}

Status TensorHandle::Tensor(const tensorflow::Tensor** t) {
  TF_RETURN_IF_ERROR(WaitReady());

  return tensor_handle_data_->Tensor(t);
}

Status TensorHandle::TensorValue(tensorflow::TensorValue* t) {
  TF_RETURN_IF_ERROR(WaitReady());

  return tensor_handle_data_->TensorValue(t);
}

Status TensorHandle::Shape(tensorflow::TensorShape* shape) {
  TF_RETURN_IF_ERROR(WaitReady());

  return tensor_handle_data_->Shape(shape);
}

Status TensorHandle::NumDims(int* num_dims) {
  DCHECK(num_dims != nullptr);
  TF_RETURN_IF_ERROR(WaitReady());

  return tensor_handle_data_->NumDims(num_dims);
}

Status TensorHandle::Dim(int dim_index, int64* dim) {
  DCHECK(dim != nullptr);
  TF_RETURN_IF_ERROR(WaitReady());

  return tensor_handle_data_->Dim(dim_index, dim);
}

Status TensorHandle::NumElements(int64* num_elements) {
  DCHECK(num_elements != nullptr);
  TF_RETURN_IF_ERROR(WaitReady());

  return tensor_handle_data_->NumElements(num_elements);
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::RemoteAddress(int64* op_id, int32* output_num) const {
  if (!is_remote_) {
    return errors::FailedPrecondition(
        "This TensorHandle refers to a local tensor handle");
  }

  *op_id = remote_op_id_;
  *output_num = remote_output_num_;
  return Status::OK();
}
#endif

void TensorHandle::SetTensor(const tensorflow::Tensor& tensor) {
  mutex_lock l(ready_mutex_);
  DCHECK(!is_remote_) << "SetTensor is not called on remote handles.";
  DCHECK(!is_ready_) << "SetTensor is only called on non-ready handles.";

  tensor_handle_data_ = absl::make_unique<LocalTensorHandleData>(tensor);
  is_ready_ = true;
}

#if !defined(IS_MOBILE_PLATFORM)
void TensorHandle::SetRemoteShape(const TensorShape& shape) {
  mutex_lock l(ready_mutex_);
  DCHECK(is_remote_) << "SeRemoteShape is only called on remote handles.";
  DCHECK(!is_ready_) << "SetRemoteShape is only called on non-ready handles.";

  tensor_handle_data_ = absl::make_unique<RemoteTensorHandleData>(
      remote_op_id_, remote_output_num_, shape, remote_eager_client_,
      remote_context_id_, ctx_);
  is_ready_ = true;
}
#endif

Status TensorHandle::CopyToDevice(EagerContext* ctx, tensorflow::Device* dstd,
                                  TensorHandle** output) {
  const tensorflow::Tensor* src = nullptr;
  TF_RETURN_IF_ERROR(Tensor(&src));
  tensorflow::Device* srcd = (device_ == nullptr) ? ctx->HostCPU() : device_;
  bool is_same_device = (srcd == dstd) || (srcd->name() == dstd->name());
  const bool dst_cpu = dstd->tensorflow_gpu_device_info() == nullptr;
  const bool src_cpu = srcd->tensorflow_gpu_device_info() == nullptr;
  if (is_same_device) {
    *output = new tensorflow::TensorHandle(*src, dstd, ctx);
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
    *output = new tensorflow::TensorHandle(dst, dstd, ctx);
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
    *output = new tensorflow::TensorHandle(dst, dstd, ctx);
  }
  return status;
}

Device* GetResourceDevice(const Tensor& t, EagerContext* ctx) {
  if (t.dtype() != DT_RESOURCE || ctx == nullptr) {
    return nullptr;
  }
  const ResourceHandle& resource_handle = t.flat<ResourceHandle>()(0);
  const auto& map = *ctx->device_map();
  auto it = map.find(resource_handle.device());
  if (it == map.end()) {
    LOG(ERROR) << "Cannot find resource device: " << resource_handle.device()
               << ".";
    return nullptr;
  }
  return it->second;
}

string TensorHandle::DebugString() const {
  VLOG(1) << "Calling TensorHandle::DebugString() on " << this;

  if (symbolic_tensor_) {
    return absl::Substitute("TF_Output($0, $1)", symbolic_tensor_->oper,
                            symbolic_tensor_->index);
  }

  string out;
  strings::StrAppend(&out, "Device: ", device_ ? device_->DebugString() : "[]");
  // Consider supporting non-CPU tensors (when device_ is non-NULL) if needed.
  strings::StrAppend(&out, ", Tensor: ",
                     device_ ? "?" : tensor_handle_data_->DebugString(), "\n");
  return out;
}

}  // namespace tensorflow
