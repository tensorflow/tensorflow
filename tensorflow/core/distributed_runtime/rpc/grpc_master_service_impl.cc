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

#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"

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

static const char* grpcMasterService_method_names[] = {
    "/tensorflow.MasterService/CreateSession",
    "/tensorflow.MasterService/ExtendSession",
    "/tensorflow.MasterService/PartialRunSetup",
    "/tensorflow.MasterService/RunStep",
    "/tensorflow.MasterService/CloseSession",
    "/tensorflow.MasterService/ListDevices",
    "/tensorflow.MasterService/Reset",
};

std::unique_ptr<MasterService::Stub> MasterService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<MasterService::Stub> stub(new MasterService::Stub(channel));
  return stub;
}

MasterService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_CreateSession_(grpcMasterService_method_names[0],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_ExtendSession_(grpcMasterService_method_names[1],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_PartialRunSetup_(grpcMasterService_method_names[2],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel),
      rpcmethod_RunStep_(grpcMasterService_method_names[3],
                         ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CloseSession_(grpcMasterService_method_names[4],
                              ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ListDevices_(grpcMasterService_method_names[5],
                             ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_Reset_(grpcMasterService_method_names[6],
                       ::grpc::internal::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status MasterService::Stub::CreateSession(
    ::grpc::ClientContext* context, const CreateSessionRequest& request,
    CreateSessionResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CreateSession_, context, request, response);
}

::grpc::Status MasterService::Stub::ExtendSession(
    ::grpc::ClientContext* context, const ExtendSessionRequest& request,
    ExtendSessionResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ExtendSession_, context, request, response);
}

::grpc::Status MasterService::Stub::PartialRunSetup(
    ::grpc::ClientContext* context, const PartialRunSetupRequest& request,
    PartialRunSetupResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_PartialRunSetup_, context, request, response);
}

::grpc::Status MasterService::Stub::RunStep(::grpc::ClientContext* context,
                                            const RunStepRequest& request,
                                            RunStepResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_RunStep_,
                                             context, request, response);
}

::grpc::Status MasterService::Stub::CloseSession(
    ::grpc::ClientContext* context, const CloseSessionRequest& request,
    CloseSessionResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CloseSession_, context, request, response);
}

::grpc::Status MasterService::Stub::ListDevices(
    ::grpc::ClientContext* context, const ListDevicesRequest& request,
    ListDevicesResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ListDevices_, context, request, response);
}

::grpc::Status MasterService::Stub::Reset(::grpc::ClientContext* context,
                                          const ResetRequest& request,
                                          ResetResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_Reset_,
                                             context, request, response);
}

MasterService::AsyncService::AsyncService() {
  for (int i = 0; i < 7; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcMasterService_method_names[i],
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

MasterService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
