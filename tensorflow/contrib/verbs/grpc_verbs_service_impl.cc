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

#include "tensorflow/contrib/verbs/grpc_verbs_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"

namespace tensorflow {

namespace grpc {

static const char* grpcVerbsService_method_names[] = {
    "/tensorflow.VerbsService/GetRemoteAddress",
};

std::unique_ptr<VerbsService::Stub> VerbsService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<VerbsService::Stub> stub(new VerbsService::Stub(channel));
  return stub;
}

VerbsService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_GetRemoteAddress_(grpcVerbsService_method_names[0],
                                  ::grpc::internal::RpcMethod::NORMAL_RPC,
                                  channel) {}

::grpc::Status VerbsService::Stub::GetRemoteAddress(
    ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
    GetRemoteAddressResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_GetRemoteAddress_, context, request, response);
}

VerbsService::AsyncService::AsyncService() {
  for (int i = 0; i < 1; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcVerbsService_method_names[i],
        ::grpc::internal::RpcMethod::NORMAL_RPC,
        nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

VerbsService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
