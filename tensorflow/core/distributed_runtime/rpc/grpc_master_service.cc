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

// GrpcMasterService implements the RPC service MasterService.
//
// A GrpcMasterService maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A GrpcMasterService knows ahead of time local devices available as
// client devices.
//
// A GrpcMasterService discovers remote devices in the background and
// keeps track of statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"

#include <string>

#include "grpcpp/alarm.h"
#include "grpcpp/server_builder.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

class GrpcMasterService : public tsl::AsyncServiceInterface {
 public:
  GrpcMasterService(Master* master, const ConfigProto& default_session_config,
                    ::grpc::ServerBuilder* builder)
      : master_impl_(master),
        is_shutdown_(false),
        default_session_config_(default_session_config) {
    builder->RegisterService(&master_service_);
    // gRPC by default will cancel requests that sit in a completion queue for
    // more than 30s. See
    // https://github.com/grpc/grpc/blob/e52e48b7ef83feeff56ed0894ce39841ea8bd483/include/grpc/impl/channel_arg_names.h#L106-L111
    // Extending this to 1 hour for Tensorflow since some graphs may have
    // periods of heavy load which may cause the server to run into these
    // cancellations.
    builder->AddChannelArgument("grpc.server_max_unrequested_time_in_server",
                                3600);
    cq_ = builder->AddCompletionQueue();
  }

  ~GrpcMasterService() override { delete shutdown_alarm_; }

  void Shutdown() override {
    bool did_shutdown = false;
    {
      mutex_lock l(mu_);
      if (!is_shutdown_) {
        LOG(INFO) << "Shutting down GrpcMasterService.";
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      // NOTE(mrry): This enqueues a special event (with a null tag)
      // that causes the completion queue to be shut down on the
      // polling thread.
      shutdown_alarm_ =
          new ::grpc::Alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(RunStep);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                              \
  do {                                                                        \
    mutex_lock l(mu_);                                                        \
    if (!is_shutdown_) {                                                      \
      tsl::Call<GrpcMasterService, grpc::MasterService::AsyncService,         \
                method##Request, method##Response>::                          \
          EnqueueRequest(&master_service_, cq_.get(),                         \
                         &grpc::MasterService::AsyncService::Request##method, \
                         &GrpcMasterService::method##Handler,                 \
                         (supports_cancel));                                  \
    }                                                                         \
  } while (0)

  void HandleRPCsLoop() override {
    ENQUEUE_REQUEST(CreateSession, true);
    ENQUEUE_REQUEST(ExtendSession, false);
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(PartialRunSetup, false);
      ENQUEUE_REQUEST(RunStep, true);
    }
    ENQUEUE_REQUEST(CloseSession, false);
    ENQUEUE_REQUEST(ListDevices, false);
    ENQUEUE_REQUEST(Reset, false);
    ENQUEUE_REQUEST(MakeCallable, false);
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(RunCallable, true);
    }
    ENQUEUE_REQUEST(ReleaseCallable, false);

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      tsl::UntypedCall<GrpcMasterService>::Tag* callback_tag =
          static_cast<tsl::UntypedCall<GrpcMasterService>::Tag*>(tag);
      if (callback_tag) {
        callback_tag->OnCompleted(this, ok);
      } else {
        // NOTE(mrry): A null `callback_tag` indicates that this is
        // the shutdown alarm.
        cq_->Shutdown();
      }
    }
  }

 private:
  Master* master_impl_ = nullptr;  // Not owned.
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  mutex mu_;
  bool is_shutdown_ TF_GUARDED_BY(mu_);
  const ConfigProto default_session_config_;
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template <class RequestMessage, class ResponseMessage>
  using MasterCall =
      tsl::Call<GrpcMasterService, grpc::MasterService::AsyncService,
                RequestMessage, ResponseMessage>;

  // RPC handler for creating a session.
  void CreateSessionHandler(
      MasterCall<CreateSessionRequest, CreateSessionResponse>* call) {
    CreateSessionRequest* rewritten_req = new CreateSessionRequest;
    rewritten_req->mutable_config()->MergeFrom(default_session_config_);
    rewritten_req->MergeFrom(call->request);
    master_impl_->CreateSession(
        rewritten_req, &call->response,
        [call, rewritten_req](const absl::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
          delete rewritten_req;
        });
    ENQUEUE_REQUEST(CreateSession, true);
  }

  // RPC handler for extending a session.
  void ExtendSessionHandler(
      MasterCall<ExtendSessionRequest, ExtendSessionResponse>* call) {
    master_impl_->ExtendSession(&call->request, &call->response,
                                [call](const absl::Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                });
    ENQUEUE_REQUEST(ExtendSession, false);
  }

  // RPC handler for setting up a partial run call.
  void PartialRunSetupHandler(
      MasterCall<PartialRunSetupRequest, PartialRunSetupResponse>* call) {
    master_impl_->PartialRunSetup(&call->request, &call->response,
                                  [call](const absl::Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(PartialRunSetup, false);
  }

  // RPC handler for running one step in a session.
  void RunStepHandler(MasterCall<RunStepRequest, RunStepResponse>* call) {
    auto* trace = TraceRpc("RunStep/Server", call->client_metadata());
    CallOptions* call_opts = new CallOptions;
    if (call->request.options().timeout_in_ms() > 0) {
      call_opts->SetTimeout(call->request.options().timeout_in_ms());
    } else {
      call_opts->SetTimeout(default_session_config_.operation_timeout_in_ms());
    }
    RunStepRequestWrapper* wrapped_request =
        new ProtoRunStepRequest(&call->request);
    MutableRunStepResponseWrapper* wrapped_response =
        new NonOwnedProtoRunStepResponse(&call->response);
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunStep(
        call_opts, wrapped_request, wrapped_response,
        [call, call_opts, wrapped_request, wrapped_response,
         trace](const absl::Status& status) {
          call->ClearCancelCallback();
          delete call_opts;
          delete wrapped_request;
          delete wrapped_response;
          delete trace;
          if (call->request.store_errors_in_response_body() && !status.ok()) {
            call->response.set_status_code(
                static_cast<error::Code>(status.code()));
            call->response.set_status_error_message(
                std::string(status.message()));
            call->SendResponse(ToGrpcStatus(absl::OkStatus()));
          } else {
            call->SendResponse(ToGrpcStatus(status));
          }
        });
    ENQUEUE_REQUEST(RunStep, true);
  }

  // RPC handler for deleting a session.
  void CloseSessionHandler(
      MasterCall<CloseSessionRequest, CloseSessionResponse>* call) {
    master_impl_->CloseSession(&call->request, &call->response,
                               [call](const absl::Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(CloseSession, false);
  }

  // RPC handler for listing devices.
  void ListDevicesHandler(
      MasterCall<ListDevicesRequest, ListDevicesResponse>* call) {
    master_impl_->ListDevices(&call->request, &call->response,
                              [call](const absl::Status& status) {
                                call->SendResponse(ToGrpcStatus(status));
                              });
    ENQUEUE_REQUEST(ListDevices, false);
  }

  // RPC handler for resetting all sessions.
  void ResetHandler(MasterCall<ResetRequest, ResetResponse>* call) {
    master_impl_->Reset(&call->request, &call->response,
                        [call](const absl::Status& status) {
                          call->SendResponse(ToGrpcStatus(status));
                        });
    ENQUEUE_REQUEST(Reset, false);
  }

  // RPC handler for making a callable.
  void MakeCallableHandler(
      MasterCall<MakeCallableRequest, MakeCallableResponse>* call) {
    master_impl_->MakeCallable(&call->request, &call->response,
                               [call](const absl::Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(MakeCallable, false);
  }

  // RPC handler for running a callable.
  void RunCallableHandler(
      MasterCall<RunCallableRequest, RunCallableResponse>* call) {
    auto* trace = TraceRpc("RunCallable/Server", call->client_metadata());
    CallOptions* call_opts = new CallOptions;
    // The timeout may be overridden by a non-zero timeout in the
    // callable's `RunOptions`; this overriding will happen inside the
    // `MasterSession` implementation.
    call_opts->SetTimeout(default_session_config_.operation_timeout_in_ms());
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunCallable(
        call_opts, &call->request, &call->response,
        [call, call_opts, trace](const absl::Status& status) {
          call->ClearCancelCallback();
          delete call_opts;
          delete trace;
          call->SendResponse(ToGrpcStatus(status));
        });
    ENQUEUE_REQUEST(RunCallable, false);
  }

  // RPC handler for making a callable.
  void ReleaseCallableHandler(
      MasterCall<ReleaseCallableRequest, ReleaseCallableResponse>* call) {
    master_impl_->ReleaseCallable(&call->request, &call->response,
                                  [call](const absl::Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(ReleaseCallable, false);
  }

#undef ENQUEUE_REQUEST

  // Start tracing, including the ID attached to the RPC.
  tsl::profiler::TraceMe* TraceRpc(
      absl::string_view name,
      const std::multimap<::grpc::string_ref, ::grpc::string_ref>& metadata) {
    absl::string_view id;
    auto it = metadata.find(GrpcIdKey());
    if (it != metadata.end()) {
      id = absl::string_view(it->second.data(), it->second.size());
    }
    return new tsl::profiler::TraceMe(
        [&] { return strings::StrCat(name, ":", id); },
        tsl::profiler::TraceMeLevel::kInfo);
  }

  GrpcMasterService(const GrpcMasterService&) = delete;
  void operator=(const GrpcMasterService&) = delete;
};

tsl::AsyncServiceInterface* NewGrpcMasterService(
    Master* master, const ConfigProto& default_session_config,
    ::grpc::ServerBuilder* builder) {
  return new GrpcMasterService(master, default_session_config, builder);
}

}  // end namespace tensorflow
