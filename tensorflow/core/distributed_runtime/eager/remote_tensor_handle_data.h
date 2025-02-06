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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Remote Tensor Handle: A handle to a Tensor on a remote host. Note that only
// the shape is known.
class RemoteTensorHandleData {
 public:
  // Constructor for lazy remote handles. A lazy remote handle is created on
  // a remote worker with an op_id and an output_num. It doesn't control the
  // lifetime of a remote handle that it refers to. If it refers to a remote
  // function input, it's sent by a client which won't serialize it until
  // the corresponding remote tensor is ready. So the remote tensor should be
  // ready when we create a lazy remote handle. If it refers to a remote output,
  // it's not ready until the shape is set.
  RemoteTensorHandleData(int64_t op_id, int output_num, uint64 context_view_id,
                         bool is_ready);
  // Constructor for unshaped remote handles. It controls the lifetime of a
  // remote handle that it refers to.
  RemoteTensorHandleData(int64_t op_id, int output_num,
                         const string& remote_task, EagerContext* ctx);
  ~RemoteTensorHandleData();

  // A remote tensor handle does not have a Tensor object, hence it can only
  // support the shape requests.
  absl::Status Shape(TensorShape* shape) const;
  absl::Status NumDims(int* num_dims) const;
  absl::Status Dim(int dim_index, int64_t* dim) const;
  absl::Status NumElements(int64_t* num_elements) const;
  absl::Status Unprotect() { return absl::OkStatus(); }

  bool IsReady() const;
  absl::Status WaitReady(const char* caller) const;
  absl::Status SetShape(const TensorShape& shape);
  absl::Status SetShapeAndRemoteTask(const TensorShape& shape,
                                     const string& remote_task);
  void Poison(absl::Status status);
  absl::Status IsPoisoned() const;

  string DebugString() const;

  // Return the op id and output num. If wait_until_ready is true, block until
  // the remote tensor is ready on a remote worker.
  absl::Status OpIdAndOutputNum(bool wait_until_ready, int64_t* op_id,
                                int32* output_num) const;

  uint64 context_view_id() const { return context_view_id_; }

 private:
  mutable mutex mu_;
  bool is_ready_ TF_GUARDED_BY(mu_);
  absl::Status is_poisoned_ TF_GUARDED_BY(mu_);
  TensorShape shape_ TF_GUARDED_BY(mu_);

  // IDs required when this class is representing a remote tensor handle.
  const int64_t op_id_;
  const int32 output_num_;
  string remote_task_ TF_GUARDED_BY(mu_);
  uint64 context_id_;
  uint64 context_view_id_;
  EagerContext* ctx_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_
