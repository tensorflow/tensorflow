/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CLIENT_CQ_TAG_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CLIENT_CQ_TAG_H_

#include "grpc++/grpc++.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// Represents a pending asynchronous client call as a tag that can be
// stored in a `grpc::CompletionQueue`.
class GrpcClientCQTag {
 public:
  GrpcClientCQTag(::grpc::ClientContext* context, StatusCallback cb)
      : context_(context), cb_(cb) {}
  ~GrpcClientCQTag() { delete context_; }

  void OnCompleted(bool ok) {
    if (!ok) {
      VLOG(2) << "Call returned with non-ok status: "
              << status_.error_message();
    }
    cb_(FromGrpcStatus(status_));
  }

  ::grpc::ClientContext* context() { return context_; }
  ::grpc::Status* status() { return &status_; }

 private:
  ::grpc::ClientContext* context_;
  ::grpc::Status status_;
  StatusCallback cb_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcClientCQTag);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CLIENT_CQ_TAG_H_
