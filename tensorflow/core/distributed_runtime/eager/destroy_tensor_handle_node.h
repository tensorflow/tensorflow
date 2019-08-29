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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_

#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// DestroyTensorHandleNode is an implementation of EagerNode which enqueues a
// request to destroy a remote tensor handle.
class DestroyTensorHandleNode : public tensorflow::EagerNode {
 public:
  DestroyTensorHandleNode(std::unique_ptr<EnqueueRequest> request,
                          EagerClient* eager_client)
      : tensorflow::EagerNode(),
        request_(std::move(request)),
        eager_client_(eager_client) {}

  Status Run() override {
    EnqueueResponse* response = new EnqueueResponse;
    return eager_client_->StreamingEnqueueAsync(
        request_.get(), response, [response](const tensorflow::Status& s) {
          if (!s.ok()) {
            LOG(WARNING) << "Ignoring an error encountered when deleting "
                            "remote tensors handles: "
                         << s.ToString();
          }
          delete response;
        });
  }

  void Abort(Status status) override {}

 private:
  std::unique_ptr<EnqueueRequest> request_;
  EagerClient* eager_client_;  // Not owned, and must outlive this node.
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_
