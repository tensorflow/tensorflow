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

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#if !defined(IS_MOBILE_PLATFORM)
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
int64_t GetRemoteDeviceIncarnation(Device* device) {
  if (device == nullptr || device->IsLocal()) return 0;
  return device->attributes().incarnation();
}

string SafeDeviceDebugString(Device* device) {
  if (device == nullptr) {
    return "[]";
  } else {
    return device->DebugString();
  }
}
}  // namespace

TensorHandle::PackedTensorHandleData::PackedTensorHandleData(
    std::vector<TensorHandle*>&& handles, const TensorShape& shape)
    : handles_(std::move(handles)), shape_(shape) {
  for (auto* handle : handles_) {
    handle->Ref();
  }
}

TensorHandle::PackedTensorHandleData::~PackedTensorHandleData() {
  for (auto* handle : handles_) {
    handle->Unref();
  }
}

Status TensorHandle::PackedTensorHandleData::Shape(TensorShape* shape) const {
  *shape = shape_;
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::NumDims(int* num_dims) const {
  *num_dims = shape_.dims();
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::Dim(int dim_index,
                                                 int64_t* dim) const {
  *dim = shape_.dim_size(dim_index);
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::NumElements(
    int64_t* num_elements) const {
  *num_elements = shape_.num_elements();
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::Unprotect() {
  for (auto* handle : handles_) {
    TF_RETURN_IF_ERROR(absl::visit([](auto& data) { return data.Unprotect(); },
                                   handle->data_));
  }
  return Status::OK();
}

bool TensorHandle::PackedTensorHandleData::IsReady() const {
  {
    tf_shared_lock l(mu_);
    if (!is_poisoned_.ok()) {
      return true;
    }
  }
  for (auto* handle : handles_) {
    if (!handle->IsReady()) {
      return false;
    }
  }
  return true;
}

Status TensorHandle::PackedTensorHandleData::WaitReady(
    const char* caller) const {
  {
    tf_shared_lock l(mu_);
    if (!is_poisoned_.ok()) {
      return is_poisoned_;
    }
  }
  for (auto* handle : handles_) {
    TF_RETURN_IF_ERROR(handle->WaitReady(caller));
  }
  return Status::OK();
}

void TensorHandle::PackedTensorHandleData::Poison(Status status) {
  mutex_lock l(mu_);
  is_poisoned_ = status;
}

string TensorHandle::PackedTensorHandleData::DebugString() const {
  string debug_str = "PackedTensorHandleData: ";
  for (const auto* handle : handles_) {
    debug_str.append(
        absl::StrCat(absl::visit([](auto& data) { return data.DebugString(); },
                                 handle->data_),
                     "; "));
  }
  return debug_str;
}

int TensorHandle::PackedTensorHandleData::NumPackedHandles() const {
  return handles_.size();
}

Status TensorHandle::PackedTensorHandleData::ExtractPackedHandle(
    const int index, TensorHandle** handle) const {
  if (index < 0 || index >= handles_.size()) {
    return errors::InvalidArgument("Expect an index within [0, ",
                                   handles_.size(), "), but got ", index);
  }
  *handle = handles_.at(index);
  return Status::OK();
}

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

  if (Type() != LOCAL) {
    *result = handle_dtypes_and_shapes_;
    return Status::OK();
  }

  // Wait for this TensorHandle to be ready.
  profiler::TraceMe activity("TensorHandle::GetResourceHandleInfo WaitReady",
                             profiler::TraceMeLevel::kVerbose);
  auto& data = absl::get<LocalTensorHandleData>(data_);
  TF_RETURN_IF_ERROR(data.WaitReady("TensorHandle::GetResourceHandleInfo"));

  *result = handle_dtypes_and_shapes_;
  return Status::OK();
}

int TensorHandle::NumPackedHandles() const {
  if (Type() != PACKED) {
    return 0;
  }
  return absl::get<PackedTensorHandleData>(data_).NumPackedHandles();
}

Status TensorHandle::ExtractPackedHandle(const int index,
                                         TensorHandle** handle) const {
  if (Type() != PACKED) {
    return errors::Internal("Invalid ExtractPackedHandleOnDevice call on a",
                            TypeString(), " handle: ", this);
  }
  return absl::get<PackedTensorHandleData>(data_).ExtractPackedHandle(index,
                                                                      handle);
}

TensorHandle* TensorHandle::CreateLocalHandle(const tensorflow::Tensor& t) {
  // TODO(b/136608821): Move away from nullptr
  tensorflow::Tensor tensor = t;
  return CreateLocalHandle(std::move(tensor),
                           /*d=*/nullptr,
                           /*op_device=*/nullptr,
                           /*ctx=*/nullptr);
}

TensorHandle* TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                              Device* op_device,
                                              EagerContext* ctx) {
  return CreateLocalHandle(std::move(t), d, op_device, nullptr, ctx);
}

TensorHandle* TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                              Device* op_device,
                                              Device* resource_device,
                                              EagerContext* ctx) {
  if (t.dtype() == DT_RESOURCE && t.NumElements() > 0) {
    return new TensorHandle(std::move(t), d, op_device, ctx);
  } else {
    return new TensorHandle(std::move(t), d, op_device, resource_device, ctx);
  }
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
                           Device* resource_device, EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(t.dtype()),
      device_((!ctx || d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(resource_device),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
                           EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(DT_RESOURCE),
      device_((!ctx || d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(
          GetResourceDevice(t.flat<class ResourceHandle>()(0), ctx)),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      handle_dtypes_and_shapes_(
          t.flat<class ResourceHandle>()(0).dtypes_and_shapes()),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}


TensorHandle* TensorHandle::CreateEmptyLocalHandle(Device* d, Device* op_device,
                                                   Device* resource_device,
                                                   tensorflow::DataType dtype,
                                                   EagerContext* ctx) {
  return new TensorHandle(d, op_device, resource_device, dtype, ctx);
}

TensorHandle::TensorHandle(Device* d, Device* op_device,
                           Device* resource_device, tensorflow::DataType dtype,
                           EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_((d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(resource_device),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<LocalTensorHandleData>) {
  DVLOG(3) << "Creating empty Local TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}

Status TensorHandle::CreatePackedHandle(std::vector<TensorHandle*>&& handles,
                                        const tensorflow::DataType dtype,
                                        const tensorflow::TensorShape& shape,
                                        const string& device_name,
                                        EagerContext* ctx,
                                        TensorHandle** packed_handle) {
  if (handles.empty()) {
    return errors::InvalidArgument("Handles should not be empty.");
  }

  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
  if (dtype == DT_RESOURCE) {
    TF_RETURN_IF_ERROR(
        handles.at(0)->GetResourceHandleDtypesAndShapes(&dtypes_and_shapes));
  }
  std::vector<string> devices;
  devices.reserve(handles.size());
  for (auto* handle : handles) {
    devices.push_back(handle->op_device() ? handle->op_device()->name()
                                          : ctx->HostCPU()->name());
  }

  CompositeDevice* composite_device = nullptr;
  TF_RETURN_IF_ERROR(ctx->FindOrCreateCompositeDevice(devices, device_name,
                                                      &composite_device));
  *packed_handle =
      new TensorHandle(std::move(handles), composite_device, dtype, shape, ctx);
  (*packed_handle)
      ->SetResourceHandleDtypeAndShape(std::move(dtypes_and_shapes));
  return Status::OK();
}

Status TensorHandle::CreatePackedHandle(std::vector<TensorHandle*>&& handles,
                                        EagerContext* ctx,
                                        TensorHandle** packed_handle) {
  if (handles.empty()) {
    return errors::InvalidArgument("Handles should not be empty.");
  }

  // Get the dtype and shape from the first handle since all handles have the
  // same dtype and shape.
  tensorflow::DataType dtype = handles.at(0)->dtype;
  tensorflow::TensorShape shape;
  TF_RETURN_IF_ERROR(handles.at(0)->Shape(&shape));
  return CreatePackedHandle(std::move(handles), dtype, shape,
                            /*device_name*/ "", ctx, packed_handle);
}

TensorHandle::TensorHandle(std::vector<TensorHandle*>&& handles, Device* device,
                           const tensorflow::DataType dtype,
                           const tensorflow::TensorShape& shape,
                           EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_(device),
      op_device_(device),
      resource_device_(dtype == DT_RESOURCE ? device : nullptr),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<PackedTensorHandleData>, std::move(handles),
            shape) {
  DVLOG(3) << "Creating a packed TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}

#if !defined(IS_MOBILE_PLATFORM)
TensorHandle* TensorHandle::CreateUnshapedRemoteHandle(
    int64_t op_id, int32_t output_num, const string& remote_task,
    tensorflow::DataType dtype, Device* d, EagerContext* ctx,
    const bool unknown_device) {
  return new TensorHandle(op_id, output_num, remote_task, dtype, d, ctx,
                          unknown_device);
}

TensorHandle::TensorHandle(int64_t op_id, int32_t output_num,
                           const string& remote_task,
                           tensorflow::DataType dtype, Device* d,
                           EagerContext* ctx, const bool unknown_device)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(dtype == DT_RESOURCE ? d : nullptr),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      unknown_device_(unknown_device),
      ctx_(ctx),
      data_(absl::in_place_type<RemoteTensorHandleData>, op_id, output_num,
            remote_task, ctx) {
  DVLOG(3) << "Creating Unshaped Remote TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}

TensorHandle* TensorHandle::CreateLazyRemoteHandle(
    int64_t op_id, int32_t output_num, tensorflow::DataType dtype, Device* d,
    const bool is_ready, EagerContext* ctx) {
  return new TensorHandle(op_id, output_num, dtype, d, is_ready, ctx);
}

TensorHandle::TensorHandle(int64_t op_id, int32_t output_num,
                           tensorflow::DataType dtype, Device* d,
                           const bool is_ready, EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(dtype == DT_RESOURCE ? d : nullptr),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<RemoteTensorHandleData>, op_id, output_num,
            ctx->GetContextViewId(), is_ready) {
  DVLOG(3) << "Creating Lazy Remote TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}
#endif

TensorHandle::~TensorHandle() { DVLOG(3) << "Deleting tensor handle " << this; }

void TensorHandle::Release() {
  DVLOG(3) << "Releasing tensor handle " << this;
  Unref();
}

tensorflow::DataType TensorHandle::DataType() const { return dtype; }

bool TensorHandle::IsReady() const {
  return absl::visit([](auto& data) { return data.IsReady(); }, data_);
}

Status TensorHandle::WaitReady(const char* caller) const {
  return absl::visit([caller](auto& data) { return data.WaitReady(caller); },
                     data_);
}

TensorHandle::HandleType TensorHandle::Type() const {
  if (data_.index() == 0) {
    return LOCAL;
  } else if (data_.index() == 1) {
    return PACKED;
  } else {
    return REMOTE;
  }
}

string TensorHandle::TypeString() const {
  if (data_.index() == 0) {
    return "LOCAL";
  } else if (data_.index() == 1) {
    return "PACKED";
  } else {
    return "REMOTE";
  }
}

Status TensorHandle::Tensor(const tensorflow::Tensor** t) const {
  DVLOG(3) << "Tensor on TensorHandle: " << this;

  if (Type() != LOCAL) {
    return errors::Internal("Invalid Tensor call on a ", TypeString(),
                            " handle: ", this);
  }

  auto& data = absl::get<LocalTensorHandleData>(data_);
  return data.Tensor(t);
}

Status TensorHandle::TensorFromDevice(const Device* d,
                                      const tensorflow::Tensor** t) const {
  DVLOG(3) << "TensorFromDevice on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    if (Type() != LOCAL) {
      return errors::Internal("Invalid Tensor call on a ", TypeString(),
                              " handle: ", this);
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

  if (d == device_) {
    if (Type() != LOCAL) {
      return errors::Internal("Invalid TensorValue call on a ", TypeString(),
                              " handle: ", this);
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

Status TensorHandle::WaitUnknownDevice() const {
  if (unknown_device_) {
    TF_RETURN_IF_ERROR(absl::visit(
        [](auto& data) {
          return data.WaitReady("TensorHandle::UnknownDevice");
        },
        data_));
  }
  return Status::OK();
}

Device* TensorHandle::DeviceOrHostCPU(const EagerContext& ctx) const {
  return (device_ == nullptr) ? ctx.HostCPU() : device_;
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
      int64_t dims;
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
  std::vector<int64_t> dims;
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

Status TensorHandle::Shape(tensorflow::PartialTensorShape* shape) const {
  DCHECK(shape != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank()) {
    *shape = inference_shape_;
    return Status::OK();
  } else {
    auto result = absl::visit(
        [](auto& data) {
          TensorShape shape;
          Status s = data.Shape(&shape);
          return std::make_pair(shape, s);
        },
        data_);
    TF_RETURN_IF_ERROR(result.second);
    *shape = result.first;
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

Status TensorHandle::Dim(int dim_index, int64_t* dim) const {
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

Status TensorHandle::NumElements(int64_t* num_elements) const {
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

  if (d == device_) {
    return absl::visit([](auto& data) { return data.Unprotect(); }, data_);
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

  if (d == device_) {
    return errors::Internal("Cannot add mirror for primary device.");
  }

  mutex_lock l(mu_);
  if (local_mirrors_.find(d) != local_mirrors_.end()) {
    return errors::AlreadyExists("Attempted to duplicate a local mirror.");
  }

  local_mirrors_.emplace(std::piecewise_construct, std::forward_as_tuple(d),
                         std::forward_as_tuple());

  return Status::OK();
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::RemoteAddress(const Device* d, const bool wait_until_ready,
                                   int64_t* op_id, int32* output_num) const {
  DVLOG(3) << "RemoteAddress on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (d != device_) {
    tf_shared_lock l(mu_);
    auto mirror = remote_mirrors_.find(d->name());
    if (mirror != remote_mirrors_.end()) {
      return mirror->second.OpIdAndOutputNum(wait_until_ready, op_id,
                                             output_num);
    }

    return errors::FailedPrecondition(
        "Could not find remote mirror for specified device");
  }

  if (Type() != REMOTE) {
    return errors::InvalidArgument("Primary device is not remote");
  }

  auto& data = absl::get<RemoteTensorHandleData>(data_);
  return data.OpIdAndOutputNum(wait_until_ready, op_id, output_num);
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

Status TensorHandle::AddUnshapedRemoteMirror(const Device* d, int64_t op_id,
                                             int output_num,
                                             const string& remote_task,
                                             EagerContext* ctx) {
  DVLOG(3) << "AddUnshapedRemoteMirror on TensorHandle: " << this
           << " device: " << d << " " << d->name() << " op_id: " << op_id
           << " output_num: " << output_num;

  mutex_lock l(mu_);
  auto remote_mirror = remote_mirrors_.find(d->name());
  if (remote_mirror != remote_mirrors_.end()) {
    if (remote_mirror->second.context_view_id() >= ctx->GetContextId()) {
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

Status TensorHandle::AddResourceShapeMirror(const Device* d, int64_t op_id,
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
      std::forward_as_tuple(op_id, output_num, ctx->GetContextViewId(),
                            /*is_ready=*/true));

  return Status::OK();
}

Status TensorHandle::SetRemoteShape(const TensorShape& shape, const Device* d,
                                    uint64 context_view_id) {
  return SetRemoteShapeAndDevice(shape, d, context_view_id, /*op_device=*/"");
}

Status TensorHandle::SetRemoteShapeAndDevice(const TensorShape& shape,
                                             const Device* d,
                                             uint64 context_view_id,
                                             string op_device) {
  DVLOG(3) << "SetRemoteShape on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (d != device_) {
    tf_shared_lock l(mu_);
    auto remote_mirror = remote_mirrors_.find(d->name());
    if (remote_mirror == remote_mirrors_.end()) {
      return Status::OK();
    }
    auto& mirror = remote_mirror->second;
    if (mirror.context_view_id() == context_view_id) {
      return mirror.SetShape(shape);
    } else if (mirror.context_view_id() < context_view_id) {
      return errors::Internal(
          absl::Substitute("Unexpected context_view_id ($0) which should not "
                           "be newer than the "
                           "one ($1) associated to the remote mirror.",
                           context_view_id, mirror.context_view_id()));
    } else {
      LOG(WARNING) << "SetRemoteShape is ignored for a remote mirror that is "
                      "accociated with a newer context_view_id.";
    }
    return Status::OK();
  }

  DCHECK(Type() == REMOTE)
      << "SetRemoteShape is only called on remote handles.";

  auto& data = absl::get<RemoteTensorHandleData>(data_);
  // context_view_id is currently used to validate mirrors. The shape of
  // RemoteTensorHandleData should be set without checking context_view_id.
  // The reason behind it is that for the primary copy of data, if the remote
  // worker / device is removed, the consumer should report a connection error
  // indicating the remote tensor is no longer available.
  // For mirrors, this is not the case because they colocate with the data
  // consuming op/function device, and we (for now) have to aggressively
  // invalidate those copies to avoid any false positives during cluster update.
  if (op_device.empty()) {
    return data.SetShape(shape);
  } else {
    if (!unknown_device_) {
      return errors::Internal("Cannot reset known devices.");
    }
    Device* device;
    TF_RETURN_IF_ERROR(ctx_->FindDeviceFromName(op_device.c_str(), &device));
    device_ = device;
    op_device_ = device;
    resource_device_ = dtype == DT_RESOURCE ? device : nullptr;
    resource_remote_device_incarnation_ =
        GetRemoteDeviceIncarnation(resource_device_);
    string remote_task;
    if (!DeviceNameUtils::GetTaskName(device->parsed_name(), &remote_task)) {
      return errors::InvalidArgument(
          "Unable to find remote task corresponding to device ",
          device->name());
    }
    return data.SetShapeAndRemoteTask(shape, remote_task);
  }
}

void TensorHandle::PoisonRemote(Status status, const Device* d,
                                uint64 context_view_id) {
  DVLOG(3) << "PoisonRemote on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (d == device_) {
    DCHECK(Type() == REMOTE)
        << "Poison can only be on remote handles: " << this;

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
  if (d == device_) {
    return errors::Internal(
        "Local mirror assign conflicts with primary device.");
  }

  mutex_lock l(mu_);
  auto elem =
      local_mirrors_.emplace(std::piecewise_construct, std::forward_as_tuple(d),
                             std::forward_as_tuple(std::move(tensor)));
  if (!elem.second) {
    return errors::AlreadyExists("Attempted to add existing mirror.");
  }

  return Status::OK();
}

Status TensorHandle::SetTensor(tensorflow::Tensor&& t, const Device* d) {
  DVLOG(3) << "SetTensor on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    DCHECK(Type() == LOCAL) << "SetTensor is not called on local handles.";

    if (t.dtype() == DT_RESOURCE && t.NumElements() > 0) {
      auto& resource_handle = t.flat<class ResourceHandle>()(0);
      handle_dtypes_and_shapes_ = resource_handle.dtypes_and_shapes();
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

  if (d == device_) {
    DCHECK(Type() != REMOTE) << "Poison can only be on local handles: " << this;
    absl::visit([status](auto& data) { data.Poison(status); }, data_);
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
                                  tensorflow::Tensor* output) const {
  tensorflow::Device* dstd = (d == nullptr) ? ctx.HostCPU() : d;
  tensorflow::Device* srcd = DeviceOrHostCPU(ctx);
  const bool dst_cpu = dstd->tensorflow_accelerator_device_info() == nullptr;
  const bool src_cpu = srcd->tensorflow_accelerator_device_info() == nullptr;
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
    src_device_context =
        srcd->tensorflow_accelerator_device_info()->default_context;
  }
  tensorflow::DeviceContext* dst_device_context = nullptr;
  if (!dst_cpu) {
    dst_device_context =
        dstd->tensorflow_accelerator_device_info()->default_context;
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
  Device* device = nullptr;
  if (!ctx->FindDeviceFromName(handle.device().c_str(), &device).ok()) {
    LOG(ERROR) << "Cannot find resource device: " << handle.device() << ".";
    return nullptr;
  }
  return device;
}

const char* TensorHandle::DeviceName(Status* status) const {
  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = op_device();
  return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                        : d->name().c_str();
}

const char* TensorHandle::BackingDeviceName(Status* status) const {
  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = device();
  return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                        : d->name().c_str();
}

const char* TensorHandle::DeviceType(Status* status) const {
  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = op_device();
  return (d == nullptr) ? "CPU" : d->parsed_name().type.c_str();
}

int TensorHandle::DeviceId(Status* status) const {
  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = op_device();
  return (d == nullptr) ? 0 : d->parsed_name().id;
}

tensorflow::ImmediateExecutionTensorHandle* TensorHandle::Copy() {
  Ref();
  return this;
}

}  // namespace tensorflow
