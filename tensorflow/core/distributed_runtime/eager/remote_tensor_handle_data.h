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

#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"

namespace tensorflow {

// Remote Tensor Handle: A handle to a Tensor on a remote host. Note that only
// the shape is known.
class RemoteTensorHandleData : public TensorHandleData {
 public:
  RemoteTensorHandleData(int64 op_id, int output_num, const TensorShape& shape,
                         eager::EagerClient* eager_client, uint64 context_id,
                         EagerContext* ctx);
  ~RemoteTensorHandleData() override;

  // A remote tensor handle does not have a Tensor object, hence it can only
  // support the shape requests.
  Status Tensor(const tensorflow::Tensor** t) const override;
  Status TensorValue(tensorflow::TensorValue* t) override;
  Status Shape(TensorShape* shape) const override;
  Status NumDims(int* num_dims) const override;
  Status Dim(int dim_index, int64* dim) const override;
  Status NumElements(int64* num_elements) const override;

  Status WaitReady() override { return Status::OK(); }

  string DebugString() const override;

  int64 op_id() const { return op_id_; }
  int32 output_num() const { return output_num_; }

 private:
  // IDs required when this class is representing a remote tensor handle.
  const int64 op_id_;
  const int32 output_num_;
  const TensorShape shape_;
  eager::EagerClient* eager_client_;
  uint64 context_id_;
  EagerContext* const ctx_;
};

// Async Remote Tensor Handle: A handle to a Tensor on a remote host. Once the
// shape has been computed this is replaced with a remote tensor handle.
class UnshapedRemoteTensorHandleData : public TensorHandleData {
 public:
  UnshapedRemoteTensorHandleData(int64 op_id, int32 output_num,
                                 uint64 shape_node_id,
                                 eager::EagerClient* eager_client,
                                 uint64 context_id, EagerContext* ctx);
  ~UnshapedRemoteTensorHandleData() override;

  // Unshaped remote tensor handles are not ready and hence cannot satisfy any
  // of these requests.
  Status Tensor(const tensorflow::Tensor** t) const override;
  Status TensorValue(tensorflow::TensorValue* t) override;
  Status Shape(TensorShape* shape) const override;
  Status NumDims(int* num_dims) const override;
  Status Dim(int dim_index, int64* dim) const override;
  Status NumElements(int64* num_elements) const override;

  // If the remote TensorShape for this handle is yet to be computed by an
  // EagerNode, this function will block till that computation is  done and the
  // handle is ready.
  Status WaitReady() override;

  string DebugString() const override;

  int64 op_id() const { return op_id_; }
  int32 output_num() const { return output_num_; }

 private:
  // IDs required when this class is representing a remote tensor handle.
  const int64 op_id_;
  const int32 output_num_;
  const uint64 shape_node_id_;
  eager::EagerClient* eager_client_;
  uint64 context_id_;
  EagerContext* const ctx_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_DATA_H_
