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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_COPY_NODE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_COPY_NODE_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace eager {

// This node supports copy a tensor:
//    Remote -> Remote
//    Local -> Remote
//    Remote -> Local
// To copy a tensor with a host, please use copy_to_device_node instead.
class RemoteCopyNode : public EagerNode {
 public:
  RemoteCopyNode(EagerContext* ctx, EagerExecutor* executor, TensorHandle* src,
                 TensorHandle* dst, Device* recv_device, uint64 recv_op_id);

  ~RemoteCopyNode() override {}

  Status Run() override;

  void Abort(Status status) override;

 private:
  Status RunSend();
  Status RunRecv();

  TensorHandle* const src_;
  TensorHandle* const dst_;
  EagerContext* const ctx_;
  EagerExecutor* const executor_;
  Device* const send_device_;
  Device* const recv_device_;
  const string wire_id_;
  const uint64 recv_op_id_;

  CancellationManager recv_cancellation_;
  Status send_status_;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_COPY_NODE_H_
