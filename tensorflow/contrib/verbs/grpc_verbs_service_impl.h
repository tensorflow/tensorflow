/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_VERBS_GRPC_VERBS_SERVICE_IMPL_H_
#define TENSORFLOW_CONTRIB_VERBS_GRPC_VERBS_SERVICE_IMPL_H_

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"

#include "tensorflow/contrib/verbs/verbs_service.pb.h"

namespace tensorflow {

namespace grpc {

// Implementation of `tensorflow.VerbsService`, based on the
// definition in "//tensorflow/contrib/verbs/verbs_service.proto",
// and the gRPC generated stub and service classes.
// See the proto file for the definition of methods and messages.
class VerbsService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status GetRemoteAddress(
        ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
        GetRemoteAddressResponse* response) = 0;
  };
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status GetRemoteAddress(
        ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
        GetRemoteAddressResponse* response) GRPC_OVERRIDE;

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    const ::grpc::internal::RpcMethod rpcmethod_GetRemoteAddress_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr< ::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestGetRemoteAddress(
        ::grpc::ServerContext* context, GetRemoteAddressRequest* request,
        ::grpc::ServerAsyncResponseWriter<GetRemoteAddressResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};

}  // namespace grpc

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_VERBS_GRPC_VERBS_SERVICE_IMPL_H_
