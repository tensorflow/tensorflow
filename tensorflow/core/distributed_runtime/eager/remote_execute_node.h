/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_

#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// EnqueueNode is an implementation of EagerNode which enqueues an operation
// via RPC in a remote EagerService.
class RemoteExecuteNode : public tensorflow::EagerNode {
 public:
  RemoteExecuteNode(tensorflow::uint64 id,
                    const tensorflow::eager::EnqueueRequest& request,
                    tensorflow::eager::EagerClient* eager_client)
      : tensorflow::EagerNode(id),
        request_(std::move(request)),
        eager_client_(eager_client) {}

  tensorflow::Status Run() override {
    tensorflow::eager::EnqueueResponse response;
    tensorflow::Status status;
    Notification n;
    eager_client_->EnqueueAsync(&request_, &response,
                                [&n, &status](const tensorflow::Status& s) {
                                  status.Update(s);
                                  n.Notify();
                                });
    n.WaitForNotification();

    return status;
  }

 private:
  EnqueueRequest request_;
  tensorflow::eager::EagerClient*
      eager_client_;  // Not owned, and must outlive the RemoteExecuteNode.
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
