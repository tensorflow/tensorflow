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

  Status TransferToClient(const TransferToClientRequest* arg,
                          TransferToClientResponse* result) override;

  Status TransferToServer(const TransferToServerRequest* arg,
                          TransferToServerResponse* result) override;

  Status TransferToInfeed(const TransferToInfeedRequest* arg,
                          TransferToInfeedResponse* result) override;

  Status TransferFromOutfeed(const TransferFromOutfeedRequest* arg,
                             TransferFromOutfeedResponse* result) override;

  Status ResetDevice(const ResetDeviceRequest* arg,
                     ResetDeviceResponse* result) override;

  Status Compile(const CompileRequest* request,
                 CompileResponse* response) override;

  Status Execute(const ExecuteRequest* request,
                 ExecuteResponse* response) override;

  Status ExecuteGraphParallel(const ExecuteGraphParallelRequest* request,
                              ExecuteParallelResponse* response) override;

  Status WaitForExecution(const WaitForExecutionRequest* arg,
                          WaitForExecutionResponse* result) override;

  Status DeconstructTuple(const DeconstructTupleRequest* arg,
                          DeconstructTupleResponse* result) override;

  Status GetComputationGraphStats(const ComputationGraphStatsRequest* request,
                                  ComputationStatsResponse* response) override;

  Status GetShape(const GetShapeRequest* arg,
                  GetShapeResponse* result) override;

  Status GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                          GetDeviceHandlesResponse* result) override;

  Status CreateChannelHandle(const CreateChannelHandleRequest* arg,
                             CreateChannelHandleResponse* result) override;

  Status ComputeConstantGraph(const ComputeConstantGraphRequest* arg,
                              ComputeConstantResponse* result) override;

  // Methods used by GlobalData.
  Status Unregister(const UnregisterRequest* arg,
                    UnregisterResponse* result) override;

  grpc::XlaService::Stub* service() { return grpc_stub_; }

 private:
  grpc::XlaService::Stub* grpc_stub_;

  TF_DISALLOW_COPY_AND_ASSIGN(GRPCStub);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_GRPC_STUB_H_
