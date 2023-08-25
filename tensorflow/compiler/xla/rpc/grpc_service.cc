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

#include "tensorflow/compiler/xla/rpc/grpc_service.h"

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_util.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<GRPCService>> GRPCService::NewService(
    se::Platform* platform) {
  std::unique_ptr<GRPCService> grpc_service(new GRPCService());
  TF_ASSIGN_OR_RETURN(grpc_service->service_,
                      ::xla::Service::NewService(platform));
  return std::move(grpc_service);
}

::grpc::Status DelegateRPC(std::function<Status()> op) {
  Status s = op();
  return tsl::ToGrpcStatus(s);
}

::grpc::Status GRPCService::Unregister(::grpc::ServerContext* context,
                                       const UnregisterRequest* arg,
                                       UnregisterResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->Unregister(arg, result); });
}

::grpc::Status GRPCService::DeconstructTuple(::grpc::ServerContext* context,
                                             const DeconstructTupleRequest* arg,
                                             DeconstructTupleResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->DeconstructTuple(arg, result);
  });
}

::grpc::Status GRPCService::GetDeviceHandles(::grpc::ServerContext* context,
                                             const GetDeviceHandlesRequest* arg,
                                             GetDeviceHandlesResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->GetDeviceHandles(arg, result);
  });
}

::grpc::Status GRPCService::Compile(::grpc::ServerContext* /*context*/,
                                    const CompileRequest* arg,
                                    CompileResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->Compile(arg, result); });
}

::grpc::Status GRPCService::Execute(::grpc::ServerContext* /*context*/,
                                    const ExecuteRequest* arg,
                                    ExecuteResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->Execute(arg, result); });
}

::grpc::Status GRPCService::ExecuteGraphParallel(
    ::grpc::ServerContext* /*context*/, const ExecuteGraphParallelRequest* arg,
    ExecuteParallelResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->ExecuteGraphParallel(arg, result);
  });
}

::grpc::Status GRPCService::WaitForExecution(::grpc::ServerContext* context,
                                             const WaitForExecutionRequest* arg,
                                             WaitForExecutionResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->WaitForExecution(arg, result);
  });
}

::grpc::Status GRPCService::TransferToClient(::grpc::ServerContext* context,
                                             const TransferToClientRequest* arg,
                                             TransferToClientResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->TransferToClient(arg, result);
  });
}

::grpc::Status GRPCService::TransferToServer(::grpc::ServerContext* context,
                                             const TransferToServerRequest* arg,
                                             TransferToServerResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->TransferToServer(arg, result);
  });
}

::grpc::Status GRPCService::TransferToInfeed(::grpc::ServerContext* context,
                                             const TransferToInfeedRequest* arg,
                                             TransferToInfeedResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->TransferToInfeed(arg, result);
  });
}

::grpc::Status GRPCService::TransferFromOutfeed(
    ::grpc::ServerContext* context, const TransferFromOutfeedRequest* arg,
    TransferFromOutfeedResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->TransferFromOutfeed(arg, result);
  });
}

::grpc::Status GRPCService::ResetDevice(::grpc::ServerContext* context,
                                        const ResetDeviceRequest* arg,
                                        ResetDeviceResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->ResetDevice(arg, result); });
}

::grpc::Status GRPCService::GetShape(::grpc::ServerContext* context,
                                     const GetShapeRequest* arg,
                                     GetShapeResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->GetShape(arg, result); });
}

}  // namespace xla
