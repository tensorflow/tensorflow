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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {
#if !defined(IS_MOBILE_PLATFORM)
const int64 kInvalidOpId = -1;
const int32 kInvalidOutputNum = -1;
#endif
}  // namespace

void TensorHandle::SetResourceHandleDtypeAndShape(
    std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes) {
  handle_dtypes_and_shapes_ = std::move(dtypes_and_shapes);
}

Status TensorHandle::GetResourceHandleDtypesAndShapes(
    std::vector<DtypeAndPartialTensorShape>* result) {
  if (dtype != DT_RESOURCE) {
    return errors::InvalidArgument(
        "TensorHandle::GetResourceDtypeAndShape should be called on tensor "
        "handles with data type DT_RESOURCE. Actual tensor: ",
        dtype);
  }

  if (IsRemote()) {
    *result = handle_dtypes_and_shapes_;
    return Status::OK();
  }

  // Wait for this TensorHandle to be ready.
  profiler::TraceMe activity(
      "TensorHandle::GetResourceHandleDtypesAndShapes WaitReady",
      profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(
      WaitReady("TensorHandle::GetResourceHandleDtypesAndShapes"));

  *result = handle_dtypes_and_shapes_;
  return Status::OK();
}

Status TensorHandle::CreateLocalHandle(const class Tensor& t,
                                       TensorHandle** h) {
  // TODO(b/136608821): Move away from nullptr
  return CreateLocalHandle(t, nullptr, nullptr, nullptr, h);
}

Status TensorHandle::CreateLocalHandle(const class Tensor& t, Device* d,
                                       EagerContext* ctx, TensorHandle** h) {
  return CreateLocalHandle(t, d, d, ctx, h);
}

Status TensorHandle::CreateLocalHandle(const class Tensor& t, Device* d,
                                       Device* op_device, EagerContext* ctx,
                                       TensorHandle** h) {
  if (t.dtype() != DT_RESOURCE) {
    *h = new TensorHandle(absl::make_unique<LocalTensorHandleData>(t),
                          t.dtype(), d, op_device, ctx);
  } else {
    const ResourceHandle& resource_handle = t.flat<class ResourceHandle>()(0);
    *h = new TensorHandle(absl::make_unique<LocalTensorHandleData>(t),
                          resource_handle, d, op_device, ctx);
  }

  return Status::OK();
}

TensorHandle::TensorHandle(std::unique_ptr<LocalTensorHandleData> t,
                           DataType dtype, Device* d, Device* op_device,
                           EagerContext* ctx)
    : dtype(dtype),
      device_(d),
      op_device_(op_device),
      resource_device_(nullptr),
#if !defined(IS_MOBILE_PLATFORM)
      remote_op_id_(kInvalidOpId),
      remote_output_num_(kInvalidOutputNum),
#endif
      ctx_(ctx),
      is_remote_(false),
      tensor_handle_data_(std::move(t)) {
  DVLOG(3) << "Creating Local TensorHandle: " << this << " device: " << device_;
  // Notify immediately since this handle is already ready.
  is_ready_notification_.Notify();
}

TensorHandle::TensorHandle(std::unique_ptr<LocalTensorHandleData> t,
                           const ResourceHandle& resource_handle, Device* d,
                           Device* op_device, EagerContext* ctx)
    : dtype(DT_RESOURCE),
      device_(d),
      op_device_(op_device),
      resource_device_(GetResourceDevice(resource_handle, ctx)),
#if !defined(IS_MOBILE_PLATFORM)
      remote_op_id_(kInvalidOpId),
      remote_output_num_(kInvalidOutputNum),
#endif
      ctx_(ctx),
      is_remote_(false),
      handle_dtypes_and_shapes_(resource_handle.dtypes_and_shapes()),
      tensor_handle_data_(std::move(t)) {
  DVLOG(3) << "Creating Local TensorHandle: " << this << " device: " << device_;
  // Notify immediately since this handle is already ready.
  is_ready_notification_.Notify();
}

Status TensorHandle::CreateAsyncLocalHandle(Device* d, Device* op_device,
                                            Device* resource_device,
                                            DataType dtype, EagerContext* ctx,
                                            TensorHandle** h) {
  *h = new TensorHandle(absl::make_unique<AsyncLocalTensorHandleData>(), d,
                        op_device, resource_device, dtype, ctx);

  return Status::OK();
}

TensorHandle::TensorHandle(std::unique_ptr<AsyncLocalTensorHandleData> t,
                           Device* d, Device* op_device,
                           Device* resource_device, DataType dtype,
                           EagerContext* ctx)
    : dtype(dtype),
      device_(d),
      op_device_(op_device),
      resource_device_(resource_device),
#if !defined(IS_MOBILE_PLATFORM)
      remote_op_id_(kInvalidOpId),
      remote_output_num_(kInvalidOutputNum),
#endif
      ctx_(ctx),
      is_remote_(false),
      tensor_handle_data_(std::move(t)) {
  DVLOG(3) << "Creating Async Local TensorHandle: " << this
           << " device: " << device_;
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::CreateRemoteHandle(
    std::unique_ptr<RemoteTensorHandleData> t, DataType dtype, Device* d,
    Device* resource_device, EagerContext* ctx, TensorHandle** h) {
  *h = new TensorHandle(std::move(t), dtype, d, resource_device, ctx);

  return Status::OK();
}

Status TensorHandle::CreateRemoteHandle(int64 op_id, int output_num,
                                        const TensorShape& shape,
                                        const string& remote_task,
                                        uint64 context_id, DataType dtype,
                                        Device* d, Device* resource_device,
                                        EagerContext* ctx, TensorHandle** h) {
  *h = new TensorHandle(
      absl::make_unique<RemoteTensorHandleData>(op_id, output_num, shape,
                                                remote_task, context_id, ctx),
      dtype, d, resource_device, ctx);
  return Status::OK();
}

TensorHandle::TensorHandle(std::unique_ptr<RemoteTensorHandleData> t,
                           DataType dtype, Device* d, Device* resource_device,
                           EagerContext* ctx)
    : dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(resource_device),
      remote_op_id_(t->op_id()),
      remote_output_num_(t->output_num()),
      ctx_(ctx),
      is_remote_(true),
      tensor_handle_data_(std::move(t)) {
  DVLOG(3) << "Creating Remote TensorHandle: " << this
           << " device: " << device_;
  // Notify immediately since this handle is already ready.
  is_ready_notification_.Notify();
}

Status TensorHandle::CreateUnshapedRemoteHandle(
    std::unique_ptr<UnshapedRemoteTensorHandleData> t, DataType dtype,
    Device* d, EagerContext* ctx, TensorHandle** h) {
  *h = new TensorHandle(std::move(t), dtype, d, ctx);

  return Status::OK();
}

Status TensorHandle::CreateUnshapedRemoteHandle(
    int64 op_id, int32 output_num, const string& remote_task, uint64 context_id,
    DataType dtype, Device* device, EagerContext* ctx, TensorHandle** h) {
  *h = new TensorHandle(absl::make_unique<UnshapedRemoteTensorHandleData>(
                            op_id, output_num, remote_task, context_id, ctx),
                        dtype, device, ctx);
  return Status::OK();
}

TensorHandle::TensorHandle(std::unique_ptr<UnshapedRemoteTensorHandleData> t,
                           DataType dtype, Device* device, EagerContext* ctx)
    : dtype(dtype),
      device_(device),
      op_device_(device),
      resource_device_(dtype == DT_RESOURCE ? device : nullptr),
      remote_op_id_(t->op_id()),
      remote_output_num_(t->output_num()),
      remote_task_(t->remote_task()),
      remote_context_id_(t->context_id()),
      ctx_(ctx),
      is_remote_(true),
      tensor_handle_data_(std::move(t)) {
  DVLOG(3) << "Creating Unshaped Remote TensorHandle: " << this
           << " device: " << device_;
}
#endif

bool TensorHandle::IsReady() {
  return is_ready_notification_.HasBeenNotified();
}

Status TensorHandle::WaitReady(const char* caller) {
  if (!IsReady()) {
    profiler::TraceMe activity(absl::StrCat(caller, " WaitReady"),
                               profiler::TraceMeLevel::kInfo);
    is_ready_notification_.WaitForNotification();
  }
  return is_poisoned_;
}

Status TensorHandle::Tensor(const tensorflow::Tensor** t) {
  TF_RETURN_IF_ERROR(WaitReady("TensorHandle::Tensor"));
  return tensor_handle_data_->Tensor(t);
}

Status TensorHandle::TensorValue(tensorflow::TensorValue* t) {
  TF_RETURN_IF_ERROR(WaitReady("TensorHandle::TensorValue"));
  return tensor_handle_data_->TensorValue(t);
}

Device* TensorHandle::DeviceOrHostCPU(EagerContext* ctx) const {
  return (device_ == nullptr) ? ctx->HostCPU() : device_;
}

Status TensorHandle::Shape(tensorflow::TensorShape* shape) {
  if (!IsReady() && inference_shape_.IsFullyDefined()) {
    bool fill = inference_shape_.AsTensorShape(shape);
    DCHECK(fill);
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(WaitReady("TensorHandle::Shape"));
    return tensor_handle_data_->Shape(shape);
  }
}

Status TensorHandle::InferenceShape(
    shape_inference::InferenceContext* const inference_context,
    shape_inference::ShapeHandle* shape_handle) {
  if (IsReady()) {
    TF_RETURN_IF_ERROR(is_poisoned_);
    std::vector<shape_inference::DimensionHandle> dims_handle;
    int num_dims;
    TF_RETURN_IF_ERROR(NumDims(&num_dims));
    for (int i = 0; i < num_dims; i++) {
      int64 dims;
      TF_RETURN_IF_ERROR(Dim(i, &dims));
      dims_handle.push_back(inference_context->MakeDim(dims));
    }
    *shape_handle = inference_context->MakeShape(dims_handle);
    return Status::OK();
  } else {
    if (inference_shape_.unknown_rank()) {
      *shape_handle = inference_context->UnknownShape();
      return Status::OK();
    }
    std::vector<shape_inference::DimensionHandle> dims_handle(
        inference_shape_.dims());
    for (int i = 0; i < dims_handle.size(); i++) {
      dims_handle[i] = inference_context->MakeDim(inference_shape_.dim_size(i));
    }
    *shape_handle = inference_context->MakeShape(dims_handle);
    return Status::OK();
  }
}

void TensorHandle::SetInferenceShape(
    shape_inference::InferenceContext* const inference_context,
    const shape_inference::ShapeHandle& shape_handle) {
  auto num_dims = inference_context->Rank(shape_handle);
  std::vector<int64> dims;
  if (num_dims == shape_inference::InferenceContext::kUnknownRank) {
    inference_shape_ = PartialTensorShape();
    return;
  }
  DCHECK_GE(num_dims, 0);
  dims.resize(num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    dims[i] = inference_context->Value(inference_context->Dim(shape_handle, i));
  }
  auto s = PartialTensorShape::MakePartialShape(dims.data(), num_dims,
                                                &inference_shape_);
  DCHECK(s.ok());
}

Status TensorHandle::CopyInferenceShape(TensorHandle* other) {
  if (IsReady()) {
    TF_RETURN_IF_ERROR(is_poisoned_);
    return Status::OK();
  }
  if (other->IsReady()) {
    TensorShape other_shape;
    TF_RETURN_IF_ERROR(other->Shape(&other_shape));
    inference_shape_ = other_shape;
  } else {
    inference_shape_ = other->inference_shape_;
  }
  return Status::OK();
}

Status TensorHandle::NumDims(int* num_dims) {
  DCHECK(num_dims != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank()) {
    *num_dims = inference_shape_.dims();
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(WaitReady("TensorHandle::NumDims"));
    return tensor_handle_data_->NumDims(num_dims);
  }
}

Status TensorHandle::Dim(int dim_index, int64* dim) {
  DCHECK(dim != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank() &&
      inference_shape_.dim_size(dim_index) != -1) {
    *dim = inference_shape_.dim_size(dim_index);
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(WaitReady("TensorHandle::Dim"));
    return tensor_handle_data_->Dim(dim_index, dim);
  }
}

Status TensorHandle::NumElements(int64* num_elements) {
  DCHECK(num_elements != nullptr);
  if (!IsReady() && inference_shape_.IsFullyDefined()) {
    *num_elements = inference_shape_.num_elements();
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(WaitReady("TensorHandle::NumElements"));
    return tensor_handle_data_->NumElements(num_elements);
  }
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::RemoteAddress(Device* d, int64* op_id,
                                   int32* output_num) const {
  if (d != device_) {
    tf_shared_lock l(mu_);
    auto mirror = remote_mirrors_.find(d);
    if (mirror != remote_mirrors_.end()) {
      *op_id = mirror->second->op_id();
      *output_num = mirror->second->output_num();
      return Status::OK();
    }

    auto unshaped_mirror = unshaped_remote_mirrors_.find(d);
    if (unshaped_mirror != unshaped_remote_mirrors_.end()) {
      *op_id = unshaped_mirror->second->op_id();
      *output_num = unshaped_mirror->second->output_num();
      return Status::OK();
    }

    return errors::FailedPrecondition(
        "Could not find remote mirror for specified device");
  }

  if (remote_op_id_ == kInvalidOpId ||
      remote_output_num_ == kInvalidOutputNum) {
    return errors::InvalidArgument("Remote handle (op_id:", remote_op_id_,
                                   ", output_num:", remote_output_num_,
                                   ") is not set.");
  }
  *op_id = remote_op_id_;
  *output_num = remote_output_num_;
  return Status::OK();
}

void TensorHandle::SetRemoteOpIdAndOutputNumToLocalTensorHandle(
    const int64 op_id, const int32 output_num) {
  DCHECK(!is_remote_);
  remote_op_id_ = op_id;
  remote_output_num_ = output_num;
}

bool TensorHandle::HasRemoteMirror(Device* d) {
  tf_shared_lock l(mu_);
  auto mirror = remote_mirrors_.find(d);
  if (mirror != remote_mirrors_.end()) {
    return true;
  }

  auto unshaped_mirror = unshaped_remote_mirrors_.find(d);
  if (unshaped_mirror != unshaped_remote_mirrors_.end()) {
    return true;
  }

  return false;
}

bool TensorHandle::HasResourceShapeMirror(Device* d) {
  tf_shared_lock l(mu_);
  auto mirror = resource_shape_mirrors_.find(d);
  if (mirror != resource_shape_mirrors_.end()) {
    return true;
  }
  return false;
}

Status TensorHandle::AddUnshapedRemoteMirror(
    std::unique_ptr<UnshapedRemoteTensorHandleData> t, Device* d) {
  mutex_lock l(mu_);
  if (remote_mirrors_.find(d) != remote_mirrors_.end()) {
    return errors::Internal("Attempted to duplicate a remote mirror.");
  }

  auto ret = unshaped_remote_mirrors_.insert(std::make_pair(d, std::move(t)));
  if (!ret.second) {
    return errors::Internal(
        "Attempted to duplicate an unshaped remote mirror.");
  }

  return Status::OK();
}

Status TensorHandle::AddResourceShapeMirror(
    std::unique_ptr<UnshapedRemoteTensorHandleData> t, Device* d) {
  mutex_lock l(mu_);
  auto ret = resource_shape_mirrors_.insert(std::make_pair(d, std::move(t)));
  if (!ret.second) {
    return errors::Internal("Attempted to duplicate a resource shape mirror.");
  }

  return Status::OK();
}

Status TensorHandle::AddRemoteMirror(std::unique_ptr<RemoteTensorHandleData> t,
                                     Device* d) {
  mutex_lock l(mu_);
  auto ret = remote_mirrors_.insert(std::make_pair(d, std::move(t)));
  if (!ret.second) {
    return errors::Internal("Attempted to duplicate a remote mirror.");
  }

  return Status::OK();
}

Status TensorHandle::SetRemoteShape(const TensorShape& shape,
                                    tensorflow::Device* d) {
  DVLOG(3) << "SetRemoteShape on TensorHandle: " << this << " device: " << d;

  if (d != device_) {
    mutex_lock l(mu_);
    if (remote_mirrors_.find(d) != remote_mirrors_.end()) {
      return errors::Internal(
          "Attempted to set remote shape for existing mirror.");
    }

    auto elem = unshaped_remote_mirrors_.find(d);
    if (elem == unshaped_remote_mirrors_.end()) {
      return errors::Internal(
          "Attempted to set remote shape for non-waiting mirror.");
    }

    auto& data = elem->second;
    data->ReleaseRemoteTensorHandle();
    remote_mirrors_[d] = absl::make_unique<RemoteTensorHandleData>(
        data->op_id(), data->output_num(), shape, data->remote_task(),
        data->context_id(), data->ctx());
    unshaped_remote_mirrors_.erase(elem);

    return Status::OK();
  }

  DCHECK(is_remote_) << "SeRemoteShape is only called on remote handles.";
  DCHECK(!is_ready_notification_.HasBeenNotified())
      << "SetRemoteShape is only called on non-ready handles.";

  UnshapedRemoteTensorHandleData* p =
      reinterpret_cast<UnshapedRemoteTensorHandleData*>(
          tensor_handle_data_.get());
  p->ReleaseRemoteTensorHandle();
  tensor_handle_data_ = absl::make_unique<RemoteTensorHandleData>(
      remote_op_id_, remote_output_num_, shape, remote_task_,
      remote_context_id_, ctx_);
  is_poisoned_ = Status::OK();
  is_ready_notification_.Notify();

  return Status::OK();
}
#endif

Status TensorHandle::SetTensor(tensorflow::Tensor&& tensor) {
  DCHECK(!is_remote_) << "SetTensor is not called on remote handles.";
  DCHECK(!is_ready_notification_.HasBeenNotified())
      << "SetTensor is only called on non-ready handles.";

  DVLOG(3) << "SetTensor on TensorHandle: " << this;

  if (tensor.dtype() == DT_RESOURCE && tensor.NumElements() > 0) {
    auto& resource_handle = tensor.flat<class ResourceHandle>()(0);
    handle_dtypes_and_shapes_ = resource_handle.dtypes_and_shapes();
  }
  tensor_handle_data_ = absl::make_unique<LocalTensorHandleData>(tensor);
  is_poisoned_ = Status::OK();
  is_ready_notification_.Notify();
  return Status::OK();
}

void TensorHandle::Poison(Status status) {
  DCHECK(!is_ready_notification_.HasBeenNotified())
      << "Poison(status) can only be called on non-ready handle: " << this;

  DVLOG(3) << "Poison on TensorHandle: " << this;

  is_poisoned_ = status;
  is_ready_notification_.Notify();
}

Status TensorHandle::CopyToDevice(EagerContext* ctx, tensorflow::Device* dstd,
                                  tensorflow::Tensor* output) {
  tensorflow::Device* srcd = DeviceOrHostCPU(ctx);
  const bool dst_cpu = dstd->tensorflow_gpu_device_info() == nullptr;
  const bool src_cpu = srcd->tensorflow_gpu_device_info() == nullptr;
  bool is_same_device =
      (srcd == dstd) || (srcd->name() == dstd->name()) || (dst_cpu && src_cpu);

  const tensorflow::Tensor* src = nullptr;
  TF_RETURN_IF_ERROR(Tensor(&src));
  if (is_same_device) {
    *output = *src;
    return Status::OK();
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
    *output = dst;
    return Status::OK();
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
    *output = dst;
    return Status::OK();
  }
  return status;
}

Device* GetResourceDevice(const ResourceHandle& handle, EagerContext* ctx) {
  if (ctx == nullptr) {
    return nullptr;
  }
  const auto& map = *ctx->device_map();
  auto it = map.find(handle.device());
  if (it == map.end()) {
    LOG(ERROR) << "Cannot find resource device: " << handle.device() << ".";
    return nullptr;
  }
  return it->second;
}

string TensorHandle::DebugString() const {
  DVLOG(1) << "Calling TensorHandle::DebugString() on " << this;

  string out;
  strings::StrAppend(&out, "Device: ", device_ ? device_->DebugString() : "[]");
  // Consider supporting non-CPU tensors (when device_ is non-NULL) if needed.
  strings::StrAppend(&out, ", Tensor: ",
                     device_ ? "?" : tensor_handle_data_->DebugString(), "\n");
  return out;
}

}  // namespace tensorflow
