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

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// DestroyTensorHandleNode is an implementation of EagerNode which enqueues a
// request to destroy a remote tensor handle.
class DestroyTensorHandleNode : public tensorflow::AsyncEagerNode {
 public:
  DestroyTensorHandleNode(std::unique_ptr<EnqueueRequest> request,
                          EagerContext* ctx, const string& remote_task,
                          bool ready)
      : tensorflow::AsyncEagerNode(),
        request_(std::move(request)),
        ctx_(ctx),
        remote_task_(remote_task),
        ready_(ready) {
    ctx_->Ref();
  }

  ~DestroyTensorHandleNode() override { ctx_->Unref(); }

  void RunAsync(StatusCallback done) override {
    auto context_id = request_->context_id();
    if (ctx_->GetContextId() != context_id) {
      // This means that this tensor was pointing to a remote device, which
      // has been changed out from under us. Simply return since there is
      // nothing we can do.
      done(Status::OK());
      return;
    }

    eager::EagerClient* eager_client;
    Status status = ctx_->GetClient(remote_task_, &eager_client);
    if (!status.ok()) {
      LOG_EVERY_N_SEC(INFO, 60)
          << "Unable to destroy remote tensor handle because the target "
          << remote_task_ << " is no longer available.";
      done(Status::OK());
      return;
    }

    EnqueueResponse* response = new EnqueueResponse;
    bool ready = ready_;
    // NOTE(fishx): Don't use StreamingEnqueueAsync here. When a
    // StreamingEnqueueAsync request fails all following requests will fail as
    // well. We don't want this request poison following requests since it is
    // safe to ignore a failing destroy tensor handle request.
    eager_client->EnqueueAsync(
        request_.get(), response,
        [response, ready, done](const tensorflow::Status& s) {
          // Omit the warning if:
          // 1. The remote tensor isn't ready.
          // 2. Lost connection to remote worker. In this case client will
          //    crash. We don't want to spam user with redundant warning logs.
          if (!s.ok() && ready && s.code() != errors::Code::UNAVAILABLE) {
            LOG_EVERY_N_SEC(WARNING, 60)
                << "Ignoring an error encountered when deleting "
                   "remote tensors handles: "
                << s.ToString();
          }
          done(Status::OK());
          delete response;
        });
  }

  void Abort(Status status) override {}

  string DebugString() const override {
    string out = "[DestroyTensorHandleNode]";
    strings::StrAppend(&out, " request: ", request_->DebugString());
    return out;
  }

 private:
  std::unique_ptr<EnqueueRequest> request_;
  EagerContext* ctx_;
  const string remote_task_;
  bool ready_;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_
