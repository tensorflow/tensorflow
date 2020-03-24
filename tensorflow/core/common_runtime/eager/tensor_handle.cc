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
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

namespace {
#if !defined(IS_MOBILE_PLATFORM)
const int64 kInvalidOpId = -1;
const int32 kInvalidOutputNum = -1;
#endif
}  // namespace

void TensorHandle::SetResourceHandleInfo(
    ResourceHandleInfo&& resource_handle_info) {
  resource_handle_info_ = std::move(resource_handle_info);
}

Status TensorHandle::GetResourceHandleInfoImpl(
    std::function<void()> set_resource_info) {
  if (dtype != DT_RESOURCE) {
    return errors::InvalidArgument(
        "TensorHandle::GetResourceDtypeAndShape should be called on tensor "
        "handles with data type DT_RESOURCE. Actual tensor: ",
        dtype);
  }

  if (IsRemote()) {
    set_resource_info();
    return Status::OK();
  }

  // Wait for this TensorHandle to be ready.
  profiler::TraceMe activity("TensorHandle::GetResourceHandleInfo WaitReady",
                             profiler::TraceMeLevel::kInfo);
  auto& data = absl::get<LocalTensorHandleData>(data_);
  TF_RETURN_IF_ERROR(data.WaitReady("TensorHandle::GetResourceHandleInfo"));

  set_resource_info();
  return Status::OK();
}

Status TensorHandle::GetResourceHandleInfo(ResourceHandleInfo* result) {
  auto get_resource_info = [result, this]() {
    *result = resource_handle_info_;
  };
  return GetResourceHandleInfoImpl(get_resource_info);
}

Status TensorHandle::GetResourceHandleDtypesAndShapes(
    std::vector<DtypeAndPartialTensorShape>* result) {
  auto get_resource_info = [result, this]() {
    *result = resource_handle_info_.dtypes_and_shapes;
  };
  return GetResourceHandleInfoImpl(get_resource_info);
}

Status TensorHandle::GetResourceAllowedDevices(std::vector<string>* result) {
  auto get_resource_info = [result, this]() {
    *result = resource_handle_info_.allowed_devices;
  };
  return GetResourceHandleInfoImpl(get_resource_info);
}

Status TensorHandle::CreateLocalHandle(const tensorflow::Tensor& t,
                                       TensorHandle** h) {
  // TODO(b/136608821): Move away from nullptr
  tensorflow::Tensor tensor = t;
  return CreateLocalHandle(std::move(tensor),
                           /*d=*/nullptr,
                           /*op_device=*/nullptr,
                           /*ctx=*/nullptr, h);
}

Status TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                       Device* op_device, EagerContext* ctx,
                                       TensorHandle** h) {
  return CreateLocalHandle(std::move(t), d, op_device, nullptr, ctx, h);
}

Status TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                       Device* op_device,
                                       Device* resource_device,
                                       EagerContext* ctx, TensorHandle** h) {
  if (t.dtype() == DT_RESOURCE && t.NumElements() > 0) {
    *h = new TensorHandle(std::move(t), d, op_device, ctx);
  } else {
    *h = new TensorHandle(std::move(t), d, op_device, resource_device, ctx);
  }

  return Status::OK();
}

Status TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, CustomDevice* d,
                                       EagerContext* ctx, TensorHandle** h) {
  *h = new TensorHandle(std::move(t), d, ctx);

  return Status::OK();
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
                           Device* resource_device, EagerContext* ctx)
    : dtype(t.dtype()),
      device_((!ctx || d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(resource_device),
      ctx_(ctx),
      implicit_mirroring_(true),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " device: " << VariantDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
                           EagerContext* ctx)
    : dtype(DT_RESOURCE),
      device_((!ctx || d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(
          GetResourceDevice(t.flat<class ResourceHandle>()(0), ctx)),
      ctx_(ctx),
      implicit_mirroring_(true),
      resource_handle_info_(
          {t.flat<class ResourceHandle>()(0).dtypes_and_shapes(),
           t.flat<class ResourceHandle>()(0).allowed_devices()}),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " device: " << VariantDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, CustomDevice* d,
                           EagerContext* ctx)
    : dtype(t.dtype()),
      device_(d),
      op_device_(nullptr),
      resource_device_(nullptr),
      ctx_(ctx),
      implicit_mirroring_(true),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
  // TODO(allenl): Figure out a better op_device story for custom devices,
  // since always setting it to CPU=nullptr doesn't make much sense.
  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " custom device: " << VariantDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}

Status TensorHandle::CreateEmptyLocalHandle(Device* d, Device* op_device,
                                            Device* resource_device,
                                            DataType dtype, EagerContext* ctx,
                                            TensorHandle** h) {
  *h = new TensorHandle(d, op_device, resource_device, dtype, ctx);

  return Status::OK();
}

TensorHandle::TensorHandle(Device* d, Device* op_device,
                           Device* resource_device, DataType dtype,
                           EagerContext* ctx)
    : dtype(dtype),
      device_((d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(resource_device),
      ctx_(ctx),
      implicit_mirroring_(true),
      data_(absl::in_place_type<LocalTensorHandleData>) {
  DVLOG(3) << "Creating empty Local TensorHandle: " << this
           << " device: " << VariantDeviceDebugString(device_);
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::CreateUnshapedRemoteHandle(int64 op_id, int32 output_num,
                                                const string& remote_task,
                                                DataType dtype, Device* d,
                                                EagerContext* ctx,
                                                TensorHandle** h) {
  *h = new TensorHandle(op_id, output_num, remote_task, dtype, d, ctx);

  return Status::OK();
}

TensorHandle::TensorHandle(int64 op_id, int32 output_num,
                           const string& remote_task, DataType dtype, Device* d,
                           EagerContext* ctx)
    : dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(dtype == DT_RESOURCE ? d : nullptr),
      ctx_(ctx),
      implicit_mirroring_(true),
      data_(absl::in_place_type<RemoteTensorHandleData>, op_id, output_num,
            remote_task, ctx) {
  DVLOG(3) << "Creating Unshaped Remote TensorHandle: " << this
           << " device: " << VariantDeviceDebugString(device_);
}

Status TensorHandle::CreateLazyRemoteHandle(int64 op_id, int32 output_num,
                                            DataType dtype, Device* d,
                                            EagerContext* ctx,
                                            TensorHandle** h) {
  *h = new TensorHandle(op_id, output_num, dtype, d, ctx);

  return Status::OK();
}

TensorHandle::TensorHandle(int64 op_id, int32 output_num, DataType dtype,
                           Device* d, EagerContext* ctx)
    : dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(dtype == DT_RESOURCE ? d : nullptr),
      ctx_(ctx),
      implicit_mirroring_(true),
      data_(absl::in_place_type<RemoteTensorHandleData>, op_id, output_num,
            ctx->GetContextViewId()) {
  DVLOG(3) << "Creating Lazy Remote TensorHandle: " << this
           << " device: " << VariantDeviceDebugString(device_);
}
#endif

bool TensorHandle::IsReady() const {
  return absl::visit([](auto& data) { return data.IsReady(); }, data_);
}

bool TensorHandle::IsRemote() const {
#if !defined(IS_MOBILE_PLATFORM)
  return data_.index() == 1;
#else
  return false;
#endif
}

Status TensorHandle::Tensor(const tensorflow::Tensor** t) const {
  DVLOG(3) << "Tensor on TensorHandle: " << this;

  if (IsRemote()) {
    return errors::Internal("Invalid Tensor call on remote handle: ", this);
  }

  auto& data = absl::get<LocalTensorHandleData>(data_);
  return data.Tensor(t);
}

Status TensorHandle::TensorFromDevice(const Device* d,
                                      const tensorflow::Tensor** t) const {
  DVLOG(3) << "TensorFromDevice on TensorHandle: " << this << " device: " << d;

  if (d == absl::get<Device*>(device_)) {
    if (IsRemote()) {
      return errors::Internal("Invalid Tensor call on remote handle: ", this);
    }

    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.Tensor(t);
  }

  tf_shared_lock l(mu_);
  auto elem = local_mirrors_.find(d);
  if (elem == local_mirrors_.end()) {
    return errors::Internal("Invalid device: ", d,
                            " in Tensor call to handle: ", this);
  }

  auto& mirror = elem->second;
  return mirror.Tensor(t);
}

Status TensorHandle::TensorValue(const Device* d, tensorflow::TensorValue* t) {
  DVLOG(3) << "TensorValue on TensorHandle: " << this << " device: " << d;

  if (d == absl::get<Device*>(device_)) {
    if (IsRemote()) {
      return errors::Internal("Invalid TensorValue call on remote handle: ",
                              this);
    }

    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.TensorValue(t);
  }

  tf_shared_lock l(mu_);
  auto elem = local_mirrors_.find(d);
  if (elem == local_mirrors_.end()) {
    return errors::Internal("Invalid device: ", d,
                            " in TensorValue call to handle: ", this);
  }

  auto& mirror = elem->second;
  return mirror.TensorValue(t);
}

TensorHandle::VariantDevice TensorHandle::DeviceOrHostCPU(
    const EagerContext& ctx) const {
  if (VariantDeviceIsCustom(device_)) {
    return device_;
  } else {
    Device* d = absl::get<Device*>(device_);
    return (d == nullptr) ? ctx.HostCPU() : d;
  }
}

Status TensorHandle::Shape(tensorflow::TensorShape* shape) {
  if (!IsReady() && inference_shape_.IsFullyDefined()) {
    bool fill = inference_shape_.AsTensorShape(shape);
    DCHECK(fill);
    return Status::OK();
  } else {
    return absl::visit([shape](auto& data) { return data.Shape(shape); },
                       data_);
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

Status TensorHandle::NumDims(int* num_dims) const {
  DCHECK(num_dims != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank()) {
    *num_dims = inference_shape_.dims();
    return Status::OK();
  } else {
    return absl::visit(
        [num_dims](auto& data) { return data.NumDims(num_dims); }, data_);
  }
}

Status TensorHandle::Dim(int dim_index, int64* dim) const {
  DCHECK(dim != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank() &&
      inference_shape_.dim_size(dim_index) != -1) {
    *dim = inference_shape_.dim_size(dim_index);
    return Status::OK();
  } else {
    return absl::visit(
        [dim_index, dim](auto& data) { return data.Dim(dim_index, dim); },
        data_);
  }
}

Status TensorHandle::NumElements(int64* num_elements) const {
  DCHECK(num_elements != nullptr);
  if (!IsReady() && inference_shape_.IsFullyDefined()) {
    *num_elements = inference_shape_.num_elements();
    return Status::OK();
  } else {
    return absl::visit(
        [num_elements](auto& data) { return data.NumElements(num_elements); },
        data_);
  }
}

Status TensorHandle::Unprotect(const Device* d) {
  DVLOG(3) << "Unprotect on TensorHandle: " << this << " device: " << d;

  if (d == absl::get<Device*>(device_)) {
    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.Unprotect();
  }

  tf_shared_lock l(mu_);
  auto elem = local_mirrors_.find(d);
  if (elem == local_mirrors_.end()) {
    return errors::Internal("Invalid device: ", d,
                            " in Unprotect call to handle: ", this);
  }

  // Check if the handle is non-empty
  auto& mirror = elem->second;
  return mirror.Unprotect();
}

bool TensorHandle::HasLocalMirror(const Device* d) const {
  DVLOG(3) << "HasLocalMirror on TensorHandle: " << this << " device: " << d;

  tf_shared_lock l(mu_);
  return local_mirrors_.find(d) != local_mirrors_.end();
}

Status TensorHandle::AddEmptyLocalMirror(const Device* d) {
  DVLOG(3) << "AddEmptyLocalMirror on TensorHandle: " << this
           << " device: " << d;

  if (!VariantDeviceIsCustom(device_) && d == absl::get<Device*>(device_)) {
    return errors::Internal("Cannot add mirror for primary device.");
  }

  mutex_lock l(mu_);
  if (local_mirrors_.find(d) != local_mirrors_.end()) {
    return errors::Internal("Attempted to duplicate a local mirror.");
  }

  local_mirrors_.emplace(std::piecewise_construct, std::forward_as_tuple(d),
                         std::forward_as_tuple());

  return Status::OK();
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::RemoteAddress(const Device* d, int64* op_id,
                                   int32* output_num) const {
  DVLOG(3) << "RemoteAddress on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (VariantDeviceIsCustom(device_) || d != absl::get<Device*>(device_)) {
    tf_shared_lock l(mu_);
    auto mirror = remote_mirrors_.find(d->name());
    if (mirror != remote_mirrors_.end()) {
      *op_id = mirror->second.op_id();
      *output_num = mirror->second.output_num();
      return Status::OK();
    }

    return errors::FailedPrecondition(
        "Could not find remote mirror for specified device");
  }

  if (!IsRemote()) {
    return errors::InvalidArgument("Primary device is not remote");
  }

  auto& data = absl::get<RemoteTensorHandleData>(data_);
  *op_id = data.op_id();
  *output_num = data.output_num();

  return Status::OK();
}

bool TensorHandle::HasRemoteMirror(const Device* d,
                                   uint64 context_view_id) const {
  DVLOG(3) << "HasRemoteMirror on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  tf_shared_lock l(mu_);
  auto mirror = remote_mirrors_.find(d->name());
  if (mirror != remote_mirrors_.end()) {
    // Check if mirror is stale
    if (mirror->second.context_view_id() != context_view_id) {
      return false;
    }
    return true;
  }

  return false;
}

bool TensorHandle::HasResourceShapeMirror(const Device* d,
                                          uint64 context_view_id) const {
  DVLOG(3) << "HasResourceShapeMirror on TensorHandle: " << this
           << " device: " << d << " " << d->name();

  tf_shared_lock l(mu_);
  auto mirror = resource_shape_mirrors_.find(d->name());
  if (mirror != resource_shape_mirrors_.end()) {
    // Check if mirror is stale
    if (mirror->second.context_view_id() != context_view_id) {
      return false;
    }
    return true;
  }
  return false;
}

Status TensorHandle::AddUnshapedRemoteMirror(const Device* d, int64 op_id,
                                             int output_num,
                                             const string& remote_task,
                                             EagerContext* ctx) {
  DVLOG(3) << "AddUnshapedRemoteMirror on TensorHandle: " << this
           << " device: " << d << " " << d->name() << " op_id: " << op_id
           << " output_num: " << output_num;

  mutex_lock l(mu_);
  auto remote_mirror = remote_mirrors_.find(d->name());
  if (remote_mirror != remote_mirrors_.end()) {
    if (remote_mirror->second.context_view_id() == ctx->GetContextId()) {
      return errors::Internal("Attempted to duplicate a remote mirror.");
    }
    // Remove stale mirror
    remote_mirrors_.erase(remote_mirror);
  }

  remote_mirrors_.emplace(
      std::piecewise_construct, std::forward_as_tuple(d->name()),
      std::forward_as_tuple(op_id, output_num, remote_task, ctx));

  return Status::OK();
}

Status TensorHandle::AddResourceShapeMirror(const Device* d, int64 op_id,
                                            int output_num, EagerContext* ctx) {
  DVLOG(3) << "AddResourceShapeMirror on TensorHandle: " << this;

  mutex_lock l(mu_);
  auto mirror = resource_shape_mirrors_.find(d->name());
  if (mirror != resource_shape_mirrors_.end()) {
    if (mirror->second.context_view_id() == ctx->GetContextViewId()) {
      return errors::Internal(
          "Attempted to duplicate a resource shape mirror.");
    }
    // Remove stale mirror
    resource_shape_mirrors_.erase(mirror);
  }

  resource_shape_mirrors_.emplace(
      std::piecewise_construct, std::forward_as_tuple(d->name()),
      std::forward_as_tuple(op_id, output_num, ctx->GetContextViewId()));

  return Status::OK();
}

Status TensorHandle::SetRemoteShape(const TensorShape& shape, const Device* d,
                                    uint64 context_view_id) {
  DVLOG(3) << "SetRemoteShape on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (VariantDeviceIsCustom(device_) || d != absl::get<Device*>(device_)) {
    mutex_lock l(mu_);
    auto remote_mirror = remote_mirrors_.find(d->name());
    if (remote_mirror != remote_mirrors_.end()) {
      auto& mirror = remote_mirror->second;
      if (mirror.context_view_id() == context_view_id) {
        return mirror.SetShape(shape);
      }
      remote_mirrors_.erase(remote_mirror);
    }

    return Status::OK();
  }

  DCHECK(IsRemote()) << "SetRemoteShape is only called on remote handles.";

  auto& data = absl::get<RemoteTensorHandleData>(data_);
  if (data.context_view_id() != context_view_id) {
    return errors::Internal("Attempted to set remote shape for an old handle.");
  }

  return data.SetShape(shape);
}

void TensorHandle::PoisonRemote(Status status, const Device* d,
                                uint64 context_view_id) {
  DVLOG(3) << "PoisonRemote on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (!VariantDeviceIsCustom(device_) && d == absl::get<Device*>(device_)) {
    DCHECK(IsRemote()) << "Poison can only be on remote handles: " << this;

    auto& data = absl::get<RemoteTensorHandleData>(data_);
    data.Poison(status);
  } else {
    tf_shared_lock l(mu_);
    auto mirror = remote_mirrors_.find(d->name());
    if (mirror != remote_mirrors_.end()) {
      if (mirror->second.context_view_id() == context_view_id) {
        mirror->second.Poison(status);
      }
    }
  }
}
#endif

Status TensorHandle::AddLocalMirror(tensorflow::Tensor&& tensor,
                                    const Device* d) {
  if (d == absl::get<Device*>(device_)) {
    return errors::Internal(
        "Local mirror assign conflicts with primary device.");
  }

  mutex_lock l(mu_);
  auto elem =
      local_mirrors_.emplace(std::piecewise_construct, std::forward_as_tuple(d),
                             std::forward_as_tuple(std::move(tensor)));
  if (!elem.second) {
    return errors::Internal("Attempted to set tensor for existing mirror.");
  }

  return Status::OK();
}

Status TensorHandle::SetTensor(tensorflow::Tensor&& t, const Device* d) {
  DVLOG(3) << "SetTensor on TensorHandle: " << this << " device: " << d;

  if (d == absl::get<Device*>(device_)) {
    DCHECK(!IsRemote()) << "SetTensor is not called on remote handles.";

    if (t.dtype() == DT_RESOURCE && t.NumElements() > 0) {
      auto& resource_handle = t.flat<class ResourceHandle>()(0);
      resource_handle_info_ = {resource_handle.dtypes_and_shapes(),
                               resource_handle.allowed_devices()};
    }
    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.SetTensor(std::move(t));
  } else {
    tf_shared_lock l(mu_);
    auto elem = local_mirrors_.find(d);
    if (elem == local_mirrors_.end()) {
      return errors::Internal(
          "Attempted to set tensor for non-existent local mirror.");
    }

    auto& mirror = elem->second;
    return mirror.SetTensor(std::move(t));
  }

  return Status::OK();
}

void TensorHandle::Poison(Status status, const Device* d) {
  DVLOG(3) << "Poison on TensorHandle: " << this << " device: " << d;

  if (!VariantDeviceIsCustom(device_) && d == absl::get<Device*>(device_)) {
    DCHECK(!IsRemote()) << "Poison can only be on local handles: " << this;

    auto& data = absl::get<LocalTensorHandleData>(data_);
    data.Poison(status);
  } else {
    tf_shared_lock l(mu_);
    auto elem = local_mirrors_.find(d);
    DCHECK(elem != local_mirrors_.end())
        << "Attempted to poison non-existent local mirror, handle: " << this
        << " device: " << d;

    auto& mirror = elem->second;
    mirror.Poison(status);
  }
}

Status TensorHandle::CopyToDevice(const EagerContext& ctx,
                                  tensorflow::Device* d,
                                  tensorflow::Tensor* output) {
  tensorflow::Device* dstd = (d == nullptr) ? ctx.HostCPU() : d;
  tensorflow::Device* srcd = absl::get<Device*>(DeviceOrHostCPU(ctx));
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

bool VariantDeviceIsCustom(
    absl::variant<Device*, CustomDevice*> variant_device) {
  return variant_device.index() != 0;
}

string VariantDeviceName(absl::variant<Device*, CustomDevice*> device) {
  return absl::visit([](auto* device) { return device->name(); }, device);
}

string VariantDeviceDebugString(absl::variant<Device*, CustomDevice*> device) {
  if (device == kVariantDeviceNull) {
    return "[]";
  } else if (VariantDeviceIsCustom(device)) {
    return absl::get<CustomDevice*>(device)->name();
  } else {
    return absl::get<Device*>(device)->DebugString();
  }
}

Device* GetResourceDevice(const ResourceHandle& handle, EagerContext* ctx) {
  if (ctx == nullptr) {
    return nullptr;
  }
  Device* device = nullptr;
  if (!ctx->FindDeviceFromName(handle.device().c_str(), &device).ok()) {
    LOG(ERROR) << "Cannot find resource device: " << handle.device() << ".";
    return nullptr;
  }
  return device;
}

string TensorHandle::DebugString() const {
  DVLOG(4) << "Calling TensorHandle::DebugString() on " << this;

  string out;
  string device_debug = VariantDeviceDebugString(device_);
  strings::StrAppend(&out, "Device: ", device_debug);
  bool is_cpu =
      !VariantDeviceIsCustom(device_) && device_ != kVariantDeviceNull;
  // Consider supporting non-CPU tensors and CPU tensors with a device_ set to
  // non-NULL if needed.
  strings::StrAppend(
      &out, ", Tensor: ",
      is_cpu ? absl::visit([](auto& data) { return data.DebugString(); }, data_)
             : "?",
      "\n");
  return out;
}

}  // namespace tensorflow
