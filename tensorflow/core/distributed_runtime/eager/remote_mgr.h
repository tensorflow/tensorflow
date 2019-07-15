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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_MGR_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace eager {

// This class manages the states required to setup an eager cluster.
// TODO(fishx): Move remote state from context to this class.
class RemoteMgr {
 public:
  explicit RemoteMgr(bool is_master) : is_master_(is_master) {}

  ~RemoteMgr() {
    for (const auto& entry : remote_tensor_handle_map_) {
      entry.second->Unref();
    }
  }

  bool IsMaster() { return is_master_; }

  void AddOperationOutputs(
      const gtl::ArraySlice<tensorflow::TensorHandle*> handles,
      int64 operation_id);

  Status GetTensorHandle(const RemoteTensorHandleInternal& remote_handle,
                         tensorflow::TensorHandle** handle);

  Status DeleteTensorHandle(const RemoteTensorHandleInternal& remote_handle);

  // Helper function to create monotonically increasing ids unique to this
  // context.
  uint64 NextOpId() {
    DCHECK(is_master_);
    mutex_lock l(next_id_mutex_);
    return next_op_id_++;
  }

  Status SerializeRemoteTensorHandle(TensorHandle* in, RemoteTensorHandle* out,
                                     Device* device);

  Status DeserializeRemoteTensorHandle(const RemoteTensorHandle& in,
                                       TensorHandle** out);

 private:
  bool is_master_;

  using RemoteTensorHandleMap =
      gtl::FlatMap<RemoteTensorHandleInternal, tensorflow::TensorHandle*,
                   RemoteTensorHandleInternalHash,
                   RemoteTensorHandleInternalEquals>;
  mutex remote_tensor_handle_mu_;
  // This map maintains the TensorHandles that is required by remote worker
  // in the cluster.
  RemoteTensorHandleMap remote_tensor_handle_map_
      GUARDED_BY(remote_tensor_handle_mu_);

  mutex next_id_mutex_;
  uint64 next_op_id_ GUARDED_BY(next_id_mutex_) = 1;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_MGR_H_
