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
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<GRPCService>> GRPCService::NewService(
    se::Platform* platform) {
  std::unique_ptr<GRPCService> grpc_service(new GRPCService());
  TF_ASSIGN_OR_RETURN(grpc_service->service_,
                      ::xla::Service::NewService(platform));
  return std::move(grpc_service);
}

::grpc::Status DelegateRPC(std::function<tensorflow::Status()> op) {
  tensorflow::Status s = op();
  return tensorflow::ToGrpcStatus(s);
}

::grpc::Status GRPCService::Computation(::grpc::ServerContext* context,
                                        const ComputationRequest* arg,
                                        ComputationResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->Computation(arg, result); });
}

::grpc::Status GRPCService::CreateOp(::grpc::ServerContext* context,
                                     const OpRequest* arg, OpResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->Op(arg, result); });
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

::grpc::Status GRPCService::SetReturnValue(::grpc::ServerContext* context,
                                           const SetReturnValueRequest* arg,
                                           SetReturnValueResponse* results) {
  return DelegateRPC([this, arg, results]() {
    return service_->SetReturnValue(arg, results);
  });
}

::grpc::Status GRPCService::Execute(::grpc::ServerContext* context,
                                    const ExecuteRequest* arg,
                                    ExecuteResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->Execute(arg, result); });
}

::grpc::Status GRPCService::ExecuteAsync(::grpc::ServerContext* context,
                                         const ExecuteAsyncRequest* arg,
                                         ExecuteAsyncResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->ExecuteAsync(arg, result); });
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

::grpc::Status GRPCService::IsConstant(::grpc::ServerContext* context,
                                       const IsConstantRequest* arg,
                                       IsConstantResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->IsConstant(arg, result); });
}

::grpc::Status GRPCService::ComputeConstant(::grpc::ServerContext* context,
                                            const ComputeConstantRequest* arg,
                                            ComputeConstantResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->ComputeConstant(arg, result); });
}

::grpc::Status GRPCService::GetShape(::grpc::ServerContext* context,
                                     const GetShapeRequest* arg,
                                     GetShapeResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->GetShape(arg, result); });
}

::grpc::Status GRPCService::GetComputationShape(
    ::grpc::ServerContext* context, const GetComputationShapeRequest* arg,
    GetComputationShapeResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->GetComputationShape(arg, result);
  });
}

::grpc::Status GRPCService::GetLocalShape(::grpc::ServerContext* context,
                                          const GetLocalShapeRequest* arg,
                                          GetLocalShapeResponse* result) {
  return DelegateRPC(
      [this, arg, result]() { return service_->GetLocalShape(arg, result); });
}

::grpc::Status GRPCService::GetComputationStats(
    ::grpc::ServerContext* context, const ComputationStatsRequest* arg,
    ComputationStatsResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->GetComputationStats(arg, result);
  });
}

::grpc::Status GRPCService::SnapshotComputation(
    ::grpc::ServerContext* context, const SnapshotComputationRequest* arg,
    SnapshotComputationResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->SnapshotComputation(arg, result);
  });
}

::grpc::Status GRPCService::LoadComputationSnapshot(
    ::grpc::ServerContext* context, const LoadComputationSnapshotRequest* arg,
    LoadComputationSnapshotResponse* result) {
  return DelegateRPC([this, arg, result]() {
    return service_->LoadComputationSnapshot(arg, result);
  });
}

}  // namespace xla
