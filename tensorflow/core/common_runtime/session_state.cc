/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {

Status SessionState::GetTensor(const string& handle, Tensor* tensor) {
  mutex_lock l(state_lock_);
  auto it = tensors_.find(handle);
  if (it == tensors_.end()) {
    return errors::InvalidArgument("The tensor with handle '", handle,
                                   "' is not in the session store.");
  }
  *tensor = it->second;
  return Status::OK();
}

Status SessionState::AddTensor(const string& handle, const Tensor& tensor) {
  mutex_lock l(state_lock_);
  if (!tensors_.insert({handle, tensor}).second) {
    return errors::InvalidArgument("Failed to add a tensor with handle '",
                                   handle, "' to the session store.");
  }
  return Status::OK();
}

Status SessionState::DeleteTensor(const string& handle) {
  mutex_lock l(state_lock_);
  if (tensors_.erase(handle) == 0) {
    return errors::InvalidArgument("Failed to delete a tensor with handle '",
                                   handle, "' in the session store.");
  }
  return Status::OK();
}

int64 SessionState::GetNewId() {
  mutex_lock l(state_lock_);
  return tensor_id_++;
}

Status TensorStore::AddTensor(const string& name, const TensorAndKey& tk) {
  mutex_lock l(lock_);
  if (!tensors_.insert({name, tk}).second) {
    return errors::InvalidArgument("Failed to add a tensor with name '", name,
                                   "' to the tensor store.");
  }
  return Status::OK();
}

Status TensorStore::SaveTensors(const std::vector<string>& output_names,
                                SessionState* session_state) {
  mutex_lock l(lock_);
  if (tensors_.size() != 0) {
    // Save only the tensors in output_names in the session.
    for (const string& name : output_names) {
      TensorId id(ParseTensorName(name));
      const string& op_name = id.first.ToString();
      auto it = tensors_.find(op_name);
      if (it != tensors_.end()) {
        // Save the tensor to the session state.
        string key = it->second.GetHandle(op_name);
        TF_RETURN_IF_ERROR(session_state->AddTensor(key, it->second.tensor));
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
