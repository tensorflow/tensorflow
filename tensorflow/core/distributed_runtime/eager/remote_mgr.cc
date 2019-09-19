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

#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace eager {

void RemoteMgr::AddOperationOutputs(
    const gtl::ArraySlice<tensorflow::TensorHandle*> handles,
    int64 operation_id) {
  mutex_lock l(remote_tensor_handle_mu_);
  for (int i = 0; i < handles.size(); i++) {
    // TODO(nareshmodi): Correctly handle operation_id not being unique.
    remote_tensor_handle_map_.emplace(
        RemoteTensorHandleInternal(operation_id, i), handles[i]);
  }
}

Status RemoteMgr::GetTensorHandleImpl(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
  auto iter = remote_tensor_handle_map_.find(remote_handle);
  if (iter == remote_tensor_handle_map_.end()) {
    return errors::InvalidArgument(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num);
  }

  *handle = iter->second;

  return Status::OK();
}

Status RemoteMgr::GetTensorHandle(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
  tf_shared_lock l(remote_tensor_handle_mu_);
  return GetTensorHandleImpl(remote_handle, handle);
}

Status RemoteMgr::GetRemoteTensorHandle(const tensorflow::TensorHandle* handle,
                                        int64* op_id, int32* output_num) {
  TF_RETURN_IF_ERROR(
      handle->RemoteAddress(handle->device(), op_id, output_num));
  tensorflow::TensorHandle* h;
  TF_RETURN_IF_ERROR(
      GetTensorHandleImpl(RemoteTensorHandleInternal(*op_id, *output_num), &h));
  if (handle != h) {
    return errors::Internal(
        "Found two different tensor handles with the same op_id:", *op_id,
        " and output_num:", *output_num);
  }
  return Status::OK();
}

Status RemoteMgr::DeleteTensorHandle(
    const RemoteTensorHandleInternal& remote_handle) {
  mutex_lock l(remote_tensor_handle_mu_);
  auto iter = remote_tensor_handle_map_.find(remote_handle);
  if (iter == remote_tensor_handle_map_.end()) {
    return errors::InvalidArgument(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num);
  }

  iter->second->Unref();
  remote_tensor_handle_map_.erase(iter);

  return Status::OK();
}

Status RemoteMgr::SerializeRemoteTensorHandle(TensorHandle* in,
                                              RemoteTensorHandle* out,
                                              Device* device,
                                              const string& device_name) {
  int64 op_id;
  int32 output_num;
  if (!in->RemoteAddress(device, &op_id, &output_num).ok()) {
    mutex_lock l(remote_tensor_handle_mu_);
    if (!GetRemoteTensorHandle(in, &op_id, &output_num).ok()) {
      op_id = NextOpId();
      output_num = 0;
      in->SetRemoteOpIdAndOutputNumToLocalTensorHandle(op_id, output_num);
      in->Ref();
      remote_tensor_handle_map_.emplace(
          RemoteTensorHandleInternal(op_id, output_num), in);
    }
  }
  out->Clear();
  out->set_op_id(op_id);
  out->set_output_num(output_num);
  out->set_op_device(in->op_device() ? in->op_device()->name() : "");
  out->set_device(device_name);
  out->set_dtype(in->dtype);
  return Status::OK();
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
    // copied to the local worker yet.
    const string& device_name =
        in.op_device().empty() ? in.device() : in.op_device();
    TF_RETURN_IF_ERROR(
        parent_->FindDeviceFromName(device_name.c_str(), &device));
    EagerClient* eager_client;
    TF_RETURN_IF_ERROR(parent_->GetClient(device, &eager_client));
    auto remote_handle_data = absl::make_unique<UnshapedRemoteTensorHandleData>(
        in.op_id(), in.output_num(), eager_client, parent_->GetContextId(),
        parent_);
    remote_handle_data->ReleaseRemoteTensorHandle();
    TF_RETURN_IF_ERROR(TensorHandle::CreateUnshapedRemoteHandle(
        std::move(remote_handle_data), in.dtype(), device, parent_, out));
  }

  return Status::OK();
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
    LOG(ERROR) << "EagerExecutor shutdown with error " << s.error_message();
  }
  executor_map_.erase(it);
}

}  // namespace eager
}  // namespace tensorflow
