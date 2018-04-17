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

#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace xla {

GRPCStub::~GRPCStub() = default;

tensorflow::Status MakeRPC(
    const std::function<::grpc::Status(::grpc::ClientContext*)>& rpc_method) {
  ::grpc::ClientContext context;
  ::grpc::Status s = rpc_method(&context);
  return tensorflow::FromGrpcStatus(s);
}

tensorflow::Status GRPCStub::TransferToClient(
    const TransferToClientRequest* request,
    TransferToClientResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToClient(context, *request, response);
  });
}

tensorflow::Status GRPCStub::TransferToServer(
    const TransferToServerRequest* request,
    TransferToServerResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToServer(context, *request, response);
  });
}

tensorflow::Status GRPCStub::TransferToInfeed(
    const TransferToInfeedRequest* request,
    TransferToInfeedResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToInfeed(context, *request, response);
  });
}

tensorflow::Status GRPCStub::TransferFromOutfeed(
    const TransferFromOutfeedRequest* request,
    TransferFromOutfeedResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferFromOutfeed(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ResetDevice(const ResetDeviceRequest* request,
                                         ResetDeviceResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ResetDevice(context, *request, response);
  });
}

tensorflow::Status GRPCStub::LoadComputationSnapshot(
    const LoadComputationSnapshotRequest* request,
    LoadComputationSnapshotResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->LoadComputationSnapshot(context, *request, response);
  });
}

tensorflow::Status GRPCStub::Execute(const ExecuteRequest* request,
                                     ExecuteResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Execute(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ExecuteGraph(const ExecuteGraphRequest* request,
                                          ExecuteResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteGraph(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ExecuteParallel(
    const ExecuteParallelRequest* request, ExecuteParallelResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteParallel(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ExecuteGraphParallel(
    const ExecuteGraphParallelRequest* request,
    ExecuteParallelResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteGraphParallel(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ExecuteAsync(const ExecuteAsyncRequest* request,
                                          ExecuteAsyncResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteAsync(context, *request, response);
  });
}

tensorflow::Status GRPCStub::WaitForExecution(
    const WaitForExecutionRequest* request,
    WaitForExecutionResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->WaitForExecution(context, *request, response);
  });
}

tensorflow::Status GRPCStub::DeconstructTuple(
    const DeconstructTupleRequest* request,
    DeconstructTupleResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->DeconstructTuple(context, *request, response);
  });
}

tensorflow::Status GRPCStub::GetComputationStats(
    const ComputationStatsRequest* request,
    ComputationStatsResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetComputationStats(context, *request, response);
  });
}

tensorflow::Status GRPCStub::GetComputationGraphStats(
    const ComputationGraphStatsRequest* request,
    ComputationStatsResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetComputationGraphStats(context, *request, response);
  });
}

tensorflow::Status GRPCStub::GetComputationShape(
    const GetComputationShapeRequest* request,
    GetComputationShapeResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetComputationShape(context, *request, response);
  });
}

tensorflow::Status GRPCStub::GetShape(const GetShapeRequest* request,
                                      GetShapeResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetShape(context, *request, response);
  });
}

tensorflow::Status GRPCStub::GetDeviceHandles(
    const GetDeviceHandlesRequest* request,
    GetDeviceHandlesResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetDeviceHandles(context, *request, response);
  });
}

tensorflow::Status GRPCStub::CreateChannelHandle(
    const CreateChannelHandleRequest* request,
    CreateChannelHandleResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->CreateChannelHandle(context, *request, response);
  });
}

// Methods used by ComputationBuilder.
tensorflow::Status GRPCStub::Computation(const ComputationRequest* request,
                                         ComputationResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Computation(context, *request, response);
  });
}

tensorflow::Status GRPCStub::Op(const OpRequest* request,
                                OpResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->CreateOp(context, *request, response);
  });
}

tensorflow::Status GRPCStub::GetLocalShape(const GetLocalShapeRequest* request,
                                           GetLocalShapeResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetLocalShape(context, *request, response);
  });
}

tensorflow::Status GRPCStub::SetReturnValue(
    const SetReturnValueRequest* request, SetReturnValueResponse* responses) {
  return MakeRPC([this, request, responses](::grpc::ClientContext* context) {
    return grpc_stub_->SetReturnValue(context, *request, responses);
  });
}

tensorflow::Status GRPCStub::IsConstant(const IsConstantRequest* request,
                                        IsConstantResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->IsConstant(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ComputeConstant(
    const ComputeConstantRequest* request, ComputeConstantResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ComputeConstant(context, *request, response);
  });
}

tensorflow::Status GRPCStub::ComputeConstantGraph(
    const ComputeConstantGraphRequest* request,
    ComputeConstantResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ComputeConstantGraph(context, *request, response);
  });
}

// Methods used by Computation.
tensorflow::Status GRPCStub::SnapshotComputation(
    const SnapshotComputationRequest* request,
    SnapshotComputationResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->SnapshotComputation(context, *request, response);
  });
}

// Methods used by GlobalData.
tensorflow::Status GRPCStub::Unregister(const UnregisterRequest* request,
                                        UnregisterResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Unregister(context, *request, response);
  });
}

}  // namespace xla
