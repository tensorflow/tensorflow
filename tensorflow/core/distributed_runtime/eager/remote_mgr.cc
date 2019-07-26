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

Status RemoteMgr::GetTensorHandle(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
  tf_shared_lock l(remote_tensor_handle_mu_);
  auto iter = remote_tensor_handle_map_.find(remote_handle);
  if (iter == remote_tensor_handle_map_.end()) {
    return errors::InvalidArgument(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num);
  }

  *handle = iter->second;

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
                                              Device* device) {
  // TODO(fishx): support serializing local tensor handle.
  int64 op_id;
  int32 output_num;
  TF_RETURN_IF_ERROR(in->RemoteAddress(device, &op_id, &output_num));
  out->Clear();
  out->set_op_id(op_id);
  out->set_output_num(output_num);
  return Status::OK();
}

Status RemoteMgr::DeserializeRemoteTensorHandle(const RemoteTensorHandle& in,
                                                TensorHandle** out) {
  // TODO(fishx): support the case when the remote tensor handle does not exist
  // in the map.
  TF_RETURN_IF_ERROR(GetTensorHandle(RemoteTensorHandleInternal(in), out));
  return Status::OK();
}

}  // namespace eager
}  // namespace tensorflow
