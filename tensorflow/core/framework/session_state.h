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

#ifndef TENSORFLOW_CORE_FRAMEWORK_SESSION_STATE_H_
#define TENSORFLOW_CORE_FRAMEWORK_SESSION_STATE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// The session state remembers the tensors we choose to keep across
// multiple run calls.
class SessionState {
 public:
  // Get a tensor from the session state.
  Status GetTensor(const string& handle, Tensor* tensor);

  // Store a tensor in the session state.
  Status AddTensor(const string& handle, const Tensor& tensor);

  // Delete a tensdor from the session state.
  Status DeleteTensor(const string& handle);

  int64 GetNewId();

  static const char* kTensorHandleResourceTypeName;

 private:
  mutex state_lock_;

  // For generating unique ids for tensors stored in the session.
  int64 tensor_id_ = 0;

  // The live tensors in the session. A map from tensor handle to tensor.
  std::unordered_map<string, Tensor> tensors_;
};

// The tensor store remembers the tensors we choose to keep for the
// current run call. It is available to every op kernel.
class TensorStore {
 public:
  struct TensorAndKey {
    Tensor tensor;
    int64 id;
    string device_name;

    string GetHandle(const string& tensor_name) {
      return strings::StrCat(tensor_name, ";", id, ";", device_name);
    }
  };

  // Add the named tensor to the tensor store for this run.
  Status AddTensor(const string& name, const TensorAndKey& tk);

  // Save the tensors in the tensor store of this run to the session.
  Status SaveTensors(const std::vector<string>& output_names,
                     SessionState* session_state);

  // Returns true if no tensors have been added to this store.
  bool empty() {
    mutex_lock l(lock_);
    return tensors_.empty();
  }

 private:
  mutex lock_;

  // The tensors that will be saved to session state when this run completes.
  // A map from tensor string name to tensor.
  std::unordered_map<string, TensorAndKey> tensors_ GUARDED_BY(lock_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_SESSION_STATE_H_
