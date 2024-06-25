/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {
Status WithErrorSourcePayload(Status error) {
  core::platform::ErrorSourceProto error_source_proto;
  error_source_proto.set_error_source(
      core::platform::ErrorSourceProto::EAGER_REMOTE_MGR);
  error.SetPayload(tensorflow::kErrorSource,
                   absl::Cord(error_source_proto.SerializeAsString()));
  return error;
}
}  // namespace

namespace eager {

void RemoteMgr::AddOperationOutputs(
    const absl::Span<tensorflow::TensorHandle* const> handles,
    int64_t operation_id) {
  mutex_lock l(remote_tensor_handle_mu_);
  for (int i = 0, end = handles.size(); i < end; i++) {
    // TODO(nareshmodi): Correctly handle operation_id not being unique.
    remote_tensor_handle_map_.emplace(
        RemoteTensorHandleInternal(operation_id, i), handles[i]);
  }
}

void RemoteMgr::AddOperationOutput(tensorflow::TensorHandle* handle,
                                   int64_t operation_id, int32_t output_num) {
  mutex_lock l(remote_tensor_handle_mu_);
  remote_tensor_handle_map_.emplace(
      RemoteTensorHandleInternal(operation_id, output_num), handle);
}

Status RemoteMgr::GetTensorHandleImpl(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
  auto iter = remote_tensor_handle_map_.find(remote_handle);
  if (iter == remote_tensor_handle_map_.end()) {
    // TODO(b/217820532): Fix the tensor deallocation order issue.
    std::string error_message = absl::StrCat(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num,
        ". One possible cause is that the tensor was accessed after "
        "deallocation in a distributed worker setup.");

    bool result;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE",
                                   true, &result));
    if (result) {
      std::string error_message_ext;
      absl::StrAppend(
          &error_message_ext, error_message,
          "Try setting "
          "`os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'` in "
          "your client to disable async streaming behavior to see if it fixes "
          "the problem.");
      return WithErrorSourcePayload(
          absl::InvalidArgumentError(error_message_ext));
    }
    return WithErrorSourcePayload(absl::InvalidArgumentError(error_message));
  }

  *handle = iter->second;

  return absl::OkStatus();
}

Status RemoteMgr::GetTensorHandle(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
  tf_shared_lock l(remote_tensor_handle_mu_);
  return GetTensorHandleImpl(remote_handle, handle);
}

Status RemoteMgr::GetMirroredResourceShape(
    const RemoteTensorHandleInternal& remote_handle,
    std::vector<DtypeAndPartialTensorShape>* handle) {
  tf_shared_lock l(mirrored_resource_shape_mu_);
  auto iter = mirrored_resource_shape_map_.find(remote_handle);
  if (iter == mirrored_resource_shape_map_.end()) {
    // TODO(b/217820532): Fix the tensor deallocation order issue.
    return WithErrorSourcePayload(errors::InvalidArgument(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num,
        ". One possible cause is that the tensor was accessed after "
        "deallocation in a distributed worker setup. Try setting "
        "`os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'` in "
        "your client to disable async streaming behavior to see if it fixes "
        "the problem."));
  }

  *handle = iter->second;

  return absl::OkStatus();
}

Status RemoteMgr::GetRemoteTensorHandle(const tensorflow::TensorHandle* handle,
                                        const bool wait_until_ready,
                                        int64_t* op_id, int32* output_num) {
  TF_RETURN_IF_ERROR(handle->RemoteAddress(handle->device(), wait_until_ready,
                                           op_id, output_num));
  tensorflow::TensorHandle* h;
  TF_RETURN_IF_ERROR(
      GetTensorHandleImpl(RemoteTensorHandleInternal(*op_id, *output_num), &h));
  if (handle != h) {
    return WithErrorSourcePayload(errors::Internal(
        "Found two different tensor handles with the same op_id:", *op_id,
        " and output_num:", *output_num));
  }
  return absl::OkStatus();
}

Status RemoteMgr::DeleteTensorHandle(
    const RemoteTensorHandleInternal& remote_handle) {
  {
    mutex_lock l(remote_tensor_handle_mu_);
    auto iter = remote_tensor_handle_map_.find(remote_handle);
    if (iter != remote_tensor_handle_map_.end()) {
      iter->second->Unref();
      remote_tensor_handle_map_.erase(iter);
      return absl::OkStatus();
    }
  }
  {
    mutex_lock l(mirrored_resource_shape_mu_);
    auto iter = mirrored_resource_shape_map_.find(remote_handle);
    if (iter != mirrored_resource_shape_map_.end()) {
      mirrored_resource_shape_map_.erase(iter);
      return absl::OkStatus();
    }
  }
  return WithErrorSourcePayload(errors::InvalidArgument(
      "Unable to find the relevant tensor remote_handle: Op ID: ",
      remote_handle.op_id, ", Output num: ", remote_handle.output_num));
}

Status RemoteMgr::SerializeRemoteTensorHandle(
    TensorHandle* in, const bool wait_until_ready, RemoteTensorHandle* out,
    Device* device, absl::string_view device_name,
    const bool serialize_resource_dtype_and_shape) {
  int64_t op_id;
  int32_t output_num;
  auto status =
      in->RemoteAddress(device, wait_until_ready, &op_id, &output_num);
  if (!status.ok()) {
    LOG(ERROR)
        << "Failed to get remote address for tensor handle with given device "
        << device->name() << " error " << status.message();
    tf_shared_lock l(remote_tensor_handle_mu_);
    TF_RETURN_IF_ERROR(
        GetRemoteTensorHandle(in, wait_until_ready, &op_id, &output_num));
  }
  out->Clear();
  out->set_op_id(op_id);
  out->set_output_num(output_num);
  out->set_op_device(in->op_device() ? in->op_device()->name() : "");
  out->set_device(device_name.empty()
                      ? std::string(in->DeviceOrHostCPU(*parent_)->name())
                      : std::string(device_name));
  out->set_dtype(in->dtype);
  if (serialize_resource_dtype_and_shape) {
    std::vector<DtypeAndPartialTensorShape> resource_dtypes_and_shapes;
    TF_RETURN_IF_ERROR(
        in->GetResourceHandleDtypesAndShapes(&resource_dtypes_and_shapes));
    for (const auto& dtype_and_shape : resource_dtypes_and_shapes) {
      ResourceDtypeAndShape* dtype_and_shape_proto =
          out->add_resource_dtypes_and_shapes();
      dtype_and_shape_proto->set_dtype(dtype_and_shape.dtype);
      dtype_and_shape.shape.AsProto(dtype_and_shape_proto->mutable_shape());
    }
  }
  return absl::OkStatus();
}

Status RemoteMgr::DeserializeRemoteTensorHandle(const RemoteTensorHandle& in,
                                                TensorHandle** out) {
  Device* device;
  if (parent_->local_device_mgr()->LookupDevice(in.op_device(), &device).ok() ||
      parent_->local_device_mgr()->LookupDevice(in.device(), &device).ok()) {
    TF_RETURN_IF_ERROR(GetTensorHandle(RemoteTensorHandleInternal(in), out));
    (*out)->Ref();
  } else {
    // Create a remote TensorHandle for remote tensors which have not been
    // copied to the local worker yet (e.g. remote function inputs).
    const string& device_name =
        in.op_device().empty() ? in.device() : in.op_device();
    TF_RETURN_IF_ERROR(
        parent_->FindDeviceFromName(device_name.c_str(), &device));
    *out = TensorHandle::CreateLazyRemoteHandle(in.op_id(), in.output_num(),
                                                in.dtype(), device,
                                                /*is_ready=*/true, parent_);
    std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
    if (!GetMirroredResourceShape(RemoteTensorHandleInternal(in),
                                  &dtypes_and_shapes)
             .ok()) {
      for (const auto& dtype_and_shape_proto :
           in.resource_dtypes_and_shapes()) {
        dtypes_and_shapes.push_back(DtypeAndPartialTensorShape{
            dtype_and_shape_proto.dtype(),
            TensorShape(dtype_and_shape_proto.shape())});
      }
      mutex_lock l(mirrored_resource_shape_mu_);
      mirrored_resource_shape_map_.emplace(
          RemoteTensorHandleInternal(in.op_id(), in.output_num()),
          dtypes_and_shapes);
    }
    (*out)->SetResourceHandleDtypeAndShape(std::move(dtypes_and_shapes));
  }

  return absl::OkStatus();
}

EagerExecutor& RemoteMgr::GetOrCreateExecutorForStream(uint64 stream_id) {
  mutex_lock l(executor_map_mu_);
  auto it = executor_map_.find(stream_id);
  if (it == executor_map_.end()) {
    auto it_and_bool = executor_map_.emplace(
        std::piecewise_construct, std::forward_as_tuple(stream_id),
        std::forward_as_tuple(/*async=*/true));
    DCHECK(it_and_bool.second);
    it = it_and_bool.first;
  }
  return it->second;
}

void RemoteMgr::DeleteExecutorForStream(uint64 stream_id) {
  mutex_lock l(executor_map_mu_);
  auto it = executor_map_.find(stream_id);
  if (it == executor_map_.end()) {
    return;
  }
  Status s = it->second.ShutDown();
  if (!s.ok()) {
    LOG(ERROR) << "EagerExecutor shutdown with error " << s.message();
  }
  executor_map_.erase(it);
}

}  // namespace eager
}  // namespace tensorflow
