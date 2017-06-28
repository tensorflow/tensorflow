/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_GRPC_VERBS_CLIENT_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_GRPC_VERBS_CLIENT_H_

#include "tensorflow/contrib/verbs/grpc_verbs_service_impl.h"
#include "tensorflow/contrib/verbs/verbs_service.pb.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// GrpcVerbsClient is a client that uses gRPC to talk to the Verbs service.
class GrpcVerbsClient {
 public:
  explicit GrpcVerbsClient(SharedGrpcChannelPtr client_channel)
      : stub_(grpc::VerbsService::NewStub(client_channel)) {}
  ~GrpcVerbsClient() {}

  Status GetRemoteAddress(CallOptions* call_options,
                          const GetRemoteAddressRequest* request,
                          GetRemoteAddressResponse* response);
  Status GetRemoteAddress(const GetRemoteAddressRequest* request,
                          GetRemoteAddressResponse* response);

 private:
  std::unique_ptr<grpc::VerbsService::Stub> stub_;

  void SetDeadline(::grpc::ClientContext* ctx, int64 time_in_ms);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcVerbsClient);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_GRPC_VERBS_CLIENT_H_
