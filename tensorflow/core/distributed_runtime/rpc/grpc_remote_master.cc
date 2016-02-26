/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"

#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/master_service.grpc.pb.h"

namespace tensorflow {

// GrpcRemoteMaster is an implementation of the MasterInterface
// that uses gRPC to talk to the Master service.
class GrpcRemoteMaster : public MasterInterface {
 public:
  explicit GrpcRemoteMaster(SharedGrpcChannelPtr client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {}

  ~GrpcRemoteMaster() override {}

  Status CreateSession(const CreateSessionRequest* request,
                       CreateSessionResponse* response) override {
    ::grpc::ClientContext ctx;
    return FromGrpcStatus(stub_->CreateSession(&ctx, *request, response));
  }

  Status ExtendSession(const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override {
    ::grpc::ClientContext ctx;
    return FromGrpcStatus(stub_->ExtendSession(&ctx, *request, response));
  }

  Status RunStep(const RunStepRequest* request,
                 RunStepResponse* response) override {
    ::grpc::ClientContext ctx;
    return FromGrpcStatus(stub_->RunStep(&ctx, *request, response));
  }

  Status CloseSession(const CloseSessionRequest* request,
                      CloseSessionResponse* response) override {
    ::grpc::ClientContext ctx;
    return FromGrpcStatus(stub_->CloseSession(&ctx, *request, response));
  }

  Status ListDevices(const ListDevicesRequest* request,
                     ListDevicesResponse* response) override {
    ::grpc::ClientContext ctx;
    return FromGrpcStatus(stub_->ListDevices(&ctx, *request, response));
  }

  Status Reset(const ResetRequest* request, ResetResponse* response) override {
    ::grpc::ClientContext ctx;
    return FromGrpcStatus(stub_->Reset(&ctx, *request, response));
  }

 private:
  std::unique_ptr<grpc::MasterService::Stub> stub_;
};

MasterInterface* NewGrpcMaster(SharedGrpcChannelPtr channel) {
  return new GrpcRemoteMaster(channel);
}

}  // namespace tensorflow
