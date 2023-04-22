/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h"

#include <functional>

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_callback.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/server_callback.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {
namespace tpu {

static const char* grpcTpuCompilationCacheService_method_names[] = {
#if defined(LIBTPU_ON_GCE)
    "/tensorflow.tpu.TpuCompilationCacheServiceExternal/GetTpuProgram",
#else  // LIBTPU_ON_GCE
    "/tensorflow.tpu.TpuCompilationCacheService/GetTpuProgram",
#endif  // LIBTPU_ON_GCE
};

std::unique_ptr<grpc::TpuCompilationCacheService::Stub>
grpc::TpuCompilationCacheService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr<grpc::TpuCompilationCacheService::Stub> stub(
      new grpc::TpuCompilationCacheService::Stub(channel));
  return stub;
}

grpc::TpuCompilationCacheService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_get_tpu_program_(grpcTpuCompilationCacheService_method_names[0],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel) {}

::grpc::Status grpc::TpuCompilationCacheService::Stub::GetTpuProgram(
    ::grpc::ClientContext* context, const RequestType& request,
    ResponseType* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_get_tpu_program_, context, request, response);
}

::grpc::ClientAsyncResponseReader<
    grpc::TpuCompilationCacheService::ResponseType>*
grpc::TpuCompilationCacheService::Stub::AsyncGetTpuProgramRaw(
    ::grpc::ClientContext* context, const RequestType& request,
    ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderFactory<
      ResponseType>::Create(channel_.get(), cq, rpcmethod_get_tpu_program_,
                            context, request, true);
}

::grpc::ClientAsyncResponseReader<
    grpc::TpuCompilationCacheService::ResponseType>*
grpc::TpuCompilationCacheService::Stub::PrepareAsyncGetTpuProgramRaw(
    ::grpc::ClientContext* context, const RequestType& request,
    ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderFactory<
      ResponseType>::Create(channel_.get(), cq, rpcmethod_get_tpu_program_,
                            context, request, false);
}

grpc::TpuCompilationCacheService::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      grpcTpuCompilationCacheService_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler<
          grpc::TpuCompilationCacheService::Service, RequestType, ResponseType>(
          std::mem_fn(
              &grpc::TpuCompilationCacheService::Service::GetTpuProgram),
          this)));
}

grpc::TpuCompilationCacheService::Service::~Service() {}

::grpc::Status grpc::TpuCompilationCacheService::Service::GetTpuProgram(
    ::grpc::ServerContext* context, const RequestType* request,
    ResponseType* response) {
  (void)context;
  (void)request;
  (void)response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

}  // namespace tpu
}  // namespace tensorflow
