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

#ifndef TENSORFLOW_COMPILER_XLA_RPC_GRPC_STUB_H_
#define TENSORFLOW_COMPILER_XLA_RPC_GRPC_STUB_H_

#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

class GRPCStub : public ServiceInterface {
 public:
  explicit GRPCStub(grpc::XlaService::Stub* stub) : grpc_stub_(stub) {}
  ~GRPCStub() override;

  tensorflow::Status TransferToClient(
      const TransferToClientRequest* arg,
      TransferToClientResponse* result) override;

  tensorflow::Status TransferToServer(
      const TransferToServerRequest* arg,
      TransferToServerResponse* result) override;

  tensorflow::Status TransferToInfeed(
      const TransferToInfeedRequest* arg,
      TransferToInfeedResponse* result) override;

  tensorflow::Status TransferFromOutfeed(
      const TransferFromOutfeedRequest* arg,
      TransferFromOutfeedResponse* result) override;

  tensorflow::Status ResetDevice(const ResetDeviceRequest* arg,
                                 ResetDeviceResponse* result) override;

  tensorflow::Status LoadComputationSnapshot(
      const LoadComputationSnapshotRequest* request,
      LoadComputationSnapshotResponse* result) override;

  tensorflow::Status Execute(const ExecuteRequest* arg,
                             ExecuteResponse* result) override;

  tensorflow::Status ExecuteGraph(const ExecuteGraphRequest* request,
                                  ExecuteResponse* response) override;

  tensorflow::Status ExecuteParallel(const ExecuteParallelRequest* arg,
                                     ExecuteParallelResponse* result) override;

  tensorflow::Status ExecuteGraphParallel(
      const ExecuteGraphParallelRequest* request,
      ExecuteParallelResponse* response) override;

  tensorflow::Status ExecuteAsync(const ExecuteAsyncRequest* arg,
                                  ExecuteAsyncResponse* result) override;

  tensorflow::Status WaitForExecution(
      const WaitForExecutionRequest* arg,
      WaitForExecutionResponse* result) override;

  tensorflow::Status DeconstructTuple(
      const DeconstructTupleRequest* arg,
      DeconstructTupleResponse* result) override;

  tensorflow::Status GetComputationStats(
      const ComputationStatsRequest* arg,
      ComputationStatsResponse* result) override;

  tensorflow::Status GetComputationGraphStats(
      const ComputationGraphStatsRequest* request,
      ComputationStatsResponse* response) override;

  tensorflow::Status GetComputationShape(
      const GetComputationShapeRequest* arg,
      GetComputationShapeResponse* result) override;

  tensorflow::Status GetShape(const GetShapeRequest* arg,
                              GetShapeResponse* result) override;

  tensorflow::Status GetDeviceHandles(
      const GetDeviceHandlesRequest* arg,
      GetDeviceHandlesResponse* result) override;

  tensorflow::Status CreateChannelHandle(
      const CreateChannelHandleRequest* arg,
      CreateChannelHandleResponse* result) override;

  // Methods used by ComputationBuilder.
  tensorflow::Status Computation(const ComputationRequest* arg,
                                 ComputationResponse* result) override;

  tensorflow::Status Op(const OpRequest* arg, OpResponse* result) override;
  tensorflow::Status GetLocalShape(const GetLocalShapeRequest* arg,
                                   GetLocalShapeResponse* result) override;

  tensorflow::Status SetReturnValue(const SetReturnValueRequest* arg,
                                    SetReturnValueResponse* results) override;

  tensorflow::Status IsConstant(const IsConstantRequest* arg,
                                IsConstantResponse* result) override;

  tensorflow::Status ComputeConstant(const ComputeConstantRequest* arg,
                                     ComputeConstantResponse* result) override;

  tensorflow::Status ComputeConstantGraph(
      const ComputeConstantGraphRequest* arg,
      ComputeConstantResponse* result) override;

  // Methods used by Computation.
  tensorflow::Status SnapshotComputation(
      const SnapshotComputationRequest* ag,
      SnapshotComputationResponse* result) override;

  // Methods used by GlobalData.
  tensorflow::Status Unregister(const UnregisterRequest* arg,
                                UnregisterResponse* result) override;

  grpc::XlaService::Stub* service() { return grpc_stub_; }

 private:
  grpc::XlaService::Stub* grpc_stub_;

  TF_DISALLOW_COPY_AND_ASSIGN(GRPCStub);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_GRPC_STUB_H_
