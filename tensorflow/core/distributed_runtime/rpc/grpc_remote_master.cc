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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"

#include <utility>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// GrpcRemoteMaster is an implementation of the MasterInterface
// that uses gRPC to talk to the Master service.
class GrpcRemoteMaster : public MasterInterface {
  using MasterServiceStub = grpc::MasterService::Stub;

 public:
  explicit GrpcRemoteMaster(const SharedGrpcChannelPtr& client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {}

  ~GrpcRemoteMaster() override {}

  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::CreateSession);
  }

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::ExtendSession);
  }

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::PartialRunSetup);
  }

  Status RunStep(CallOptions* call_options, RunStepRequestWrapper* request,
                 MutableRunStepResponseWrapper* response) override {
    ::grpc::ClientContext ctx;
    auto trace = TraceRpc("RunStep/Client", &ctx);
    return Call(&ctx, call_options, &request->ToProto(),
                get_proto_from_wrapper(response), &MasterServiceStub::RunStep);
  }

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::CloseSession);
  }

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::ListDevices);
  }

  Status Reset(CallOptions* call_options, const ResetRequest* request,
               ResetResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::Reset);
  }

  Status MakeCallable(CallOptions* call_options,
                      const MakeCallableRequest* request,
                      MakeCallableResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::MakeCallable);
  }
  Status RunCallable(CallOptions* call_options,
                     const RunCallableRequest* request,
                     RunCallableResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::RunCallable);
  }
  Status ReleaseCallable(CallOptions* call_options,
                         const ReleaseCallableRequest* request,
                         ReleaseCallableResponse* response) override {
    ::grpc::ClientContext ctx;
    return Call(&ctx, call_options, request, response,
                &MasterServiceStub::ReleaseCallable);
  }

 private:
  // Start tracing, attaching a unique ID to both the trace and the RPC.
  tracing::ScopedActivity TraceRpc(StringPiece name,
                                   ::grpc::ClientContext* ctx) {
    string trace_id = strings::StrCat(tracing::GetUniqueArg());
    ctx->AddMetadata(GrpcIdKey(), trace_id);
    return tracing::ScopedActivity(name, trace_id);
  }

  void SetDeadline(::grpc::ClientContext* ctx, int64 time_in_ms) {
    if (time_in_ms > 0) {
      ctx->set_deadline(gpr_time_from_millis(time_in_ms, GPR_TIMESPAN));
    }
  }

  template <typename Request, typename Response>
  Status Call(::grpc::ClientContext* ctx, CallOptions* call_options,
              const Request* request, Response* response,
              ::grpc::Status (MasterServiceStub::*pfunc)(::grpc::ClientContext*,
                                                         const Request&,
                                                         Response*)) {
    ctx->set_fail_fast(false);
    SetDeadline(ctx, call_options->GetTimeout());
    return FromGrpcStatus((stub_.get()->*pfunc)(ctx, *request, response));
  }

  std::unique_ptr<MasterServiceStub> stub_;
};

MasterInterface* NewGrpcMaster(const SharedGrpcChannelPtr& channel) {
  return new GrpcRemoteMaster(channel);
}

}  // namespace tensorflow
