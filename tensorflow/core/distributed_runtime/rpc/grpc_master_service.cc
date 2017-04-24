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

// GrpcMasterService implements the RPC service MasterSerivce.
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
// Each session analyses the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"

#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(Master* master, int64 default_timeout_in_ms,
                    ::grpc::ServerBuilder* builder)
      : master_impl_(master),
        default_timeout_in_ms_(default_timeout_in_ms),
        is_shutdown_(false) {
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
  }

  ~GrpcMasterService() {
    delete shutdown_alarm_;
  }

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
      Call<GrpcMasterService, grpc::MasterService::AsyncService,              \
           method##Request, method##Response>::                               \
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

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      UntypedCall<GrpcMasterService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcMasterService>::Tag*>(tag);
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
  const int64 default_timeout_in_ms_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template <class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  // RPC handler for creating a session.
  void CreateSessionHandler(
      MasterCall<CreateSessionRequest, CreateSessionResponse>* call) {
    master_impl_->CreateSession(&call->request, &call->response,
                                [call](const Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                });
    ENQUEUE_REQUEST(CreateSession, true);
  }

  // RPC handler for extending a session.
  void ExtendSessionHandler(
      MasterCall<ExtendSessionRequest, ExtendSessionResponse>* call) {
    master_impl_->ExtendSession(&call->request, &call->response,
                                [call](const Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                });
    ENQUEUE_REQUEST(ExtendSession, false);
  }

  // RPC handler for setting up a partial run call.
  void PartialRunSetupHandler(
      MasterCall<PartialRunSetupRequest, PartialRunSetupResponse>* call) {
    master_impl_->PartialRunSetup(&call->request, &call->response,
                                  [call](const Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(PartialRunSetup, false);
  }

  // RPC handler for running one step in a session.
  void RunStepHandler(MasterCall<RunStepRequest, RunStepResponse>* call) {
    CallOptions* call_opts = new CallOptions;
    if (call->request.options().timeout_in_ms() > 0) {
      call_opts->SetTimeout(call->request.options().timeout_in_ms());
    } else {
      call_opts->SetTimeout(default_timeout_in_ms_);
    }
    RunStepRequestWrapper* wrapped_request =
        new ProtoRunStepRequest(&call->request);
    MutableRunStepResponseWrapper* wrapped_response =
        new NonOwnedProtoRunStepResponse(&call->response);
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunStep(call_opts, wrapped_request, wrapped_response,
                          [call, call_opts, wrapped_request,
                           wrapped_response](const Status& status) {
                            call->ClearCancelCallback();
                            delete call_opts;
                            delete wrapped_request;
                            call->SendResponse(ToGrpcStatus(status));
                          });
    ENQUEUE_REQUEST(RunStep, true);
  }

  // RPC handler for deleting a session.
  void CloseSessionHandler(
      MasterCall<CloseSessionRequest, CloseSessionResponse>* call) {
    master_impl_->CloseSession(&call->request, &call->response,
                               [call](const Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(CloseSession, false);
  }

  // RPC handler for listing devices.
  void ListDevicesHandler(
      MasterCall<ListDevicesRequest, ListDevicesResponse>* call) {
    master_impl_->ListDevices(&call->request, &call->response,
                              [call](const Status& status) {
                                call->SendResponse(ToGrpcStatus(status));
                              });
    ENQUEUE_REQUEST(ListDevices, false);
  }

  // RPC handler for resetting all sessions.
  void ResetHandler(MasterCall<ResetRequest, ResetResponse>* call) {
    master_impl_->Reset(&call->request, &call->response,
                        [call](const Status& status) {
                          call->SendResponse(ToGrpcStatus(status));
                        });
    ENQUEUE_REQUEST(Reset, false);
  }
#undef ENQUEUE_REQUEST

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcMasterService);
};

AsyncServiceInterface* NewGrpcMasterService(Master* master,
                                            int64 default_timeout_in_ms,
                                            ::grpc::ServerBuilder* builder) {
  return new GrpcMasterService(master, default_timeout_in_ms, builder);
}

}  // end namespace tensorflow
