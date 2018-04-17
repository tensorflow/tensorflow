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

#ifndef TENSORFLOW_COMPILER_XLA_RPC_GRPC_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_RPC_GRPC_SERVICE_H_

#include "grpc++/server_context.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace xla {

// Service implementation which wraps a XLA Service with a GRPC interface.
class GRPCService : public grpc::XlaService::Service {
 public:
  // Factory for creating a RPCService. The parameter platform is the platform
  // that the service should target. If platform is null then the default
  // platform is used.
  static StatusOr<std::unique_ptr<GRPCService>> NewService(
      perftools::gputools::Platform* platform = nullptr);

  ::grpc::Status Computation(::grpc::ServerContext* context,
                             const ComputationRequest* arg,
                             ComputationResponse* result) override;

  ::grpc::Status CreateOp(::grpc::ServerContext* context, const OpRequest* arg,
                          OpResponse* result) override;

  ::grpc::Status Unregister(::grpc::ServerContext* context,
                            const UnregisterRequest* arg,
                            UnregisterResponse* result) override;

  ::grpc::Status DeconstructTuple(::grpc::ServerContext* context,
                                  const DeconstructTupleRequest* arg,
                                  DeconstructTupleResponse* result) override;

  ::grpc::Status SetReturnValue(::grpc::ServerContext* context,
                                const SetReturnValueRequest* arg,
                                SetReturnValueResponse* results) override;

  ::grpc::Status Execute(::grpc::ServerContext* context,
                         const ExecuteRequest* arg,
                         ExecuteResponse* result) override;

  ::grpc::Status ExecuteAsync(::grpc::ServerContext* context,
                              const ExecuteAsyncRequest* arg,
                              ExecuteAsyncResponse* result) override;

  ::grpc::Status WaitForExecution(::grpc::ServerContext* context,
                                  const WaitForExecutionRequest* arg,
                                  WaitForExecutionResponse* result) override;

  ::grpc::Status TransferToClient(::grpc::ServerContext* context,
                                  const TransferToClientRequest* arg,
                                  TransferToClientResponse* result) override;

  ::grpc::Status TransferToServer(::grpc::ServerContext* context,
                                  const TransferToServerRequest* arg,
                                  TransferToServerResponse* result) override;

  ::grpc::Status TransferToInfeed(::grpc::ServerContext* context,
                                  const TransferToInfeedRequest* arg,
                                  TransferToInfeedResponse* result) override;

  ::grpc::Status TransferFromOutfeed(
      ::grpc::ServerContext* context, const TransferFromOutfeedRequest* arg,
      TransferFromOutfeedResponse* result) override;

  ::grpc::Status ResetDevice(::grpc::ServerContext* context,
                             const ResetDeviceRequest* arg,
                             ResetDeviceResponse* result) override;

  ::grpc::Status IsConstant(::grpc::ServerContext* context,
                            const IsConstantRequest* arg,
                            IsConstantResponse* result) override;

  ::grpc::Status ComputeConstant(::grpc::ServerContext* context,
                                 const ComputeConstantRequest* arg,
                                 ComputeConstantResponse* result) override;

  ::grpc::Status GetShape(::grpc::ServerContext* context,
                          const GetShapeRequest* arg,
                          GetShapeResponse* result) override;

  ::grpc::Status GetComputationShape(
      ::grpc::ServerContext* context, const GetComputationShapeRequest* arg,
      GetComputationShapeResponse* result) override;

  ::grpc::Status GetLocalShape(::grpc::ServerContext* context,
                               const GetLocalShapeRequest* arg,
                               GetLocalShapeResponse* result) override;

  ::grpc::Status GetComputationStats(::grpc::ServerContext* context,
                                     const ComputationStatsRequest* arg,
                                     ComputationStatsResponse* result) override;

  ::grpc::Status SnapshotComputation(
      ::grpc::ServerContext* context, const SnapshotComputationRequest* arg,
      SnapshotComputationResponse* result) override;

  ::grpc::Status LoadComputationSnapshot(
      ::grpc::ServerContext* context, const LoadComputationSnapshotRequest* arg,
      LoadComputationSnapshotResponse* result) override;

 private:
  std::unique_ptr<::xla::Service> service_;

  GRPCService() {}
  GRPCService(const GRPCService&) = delete;
  void operator=(const GRPCService&) = delete;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_GRPC_SERVICE_H_
