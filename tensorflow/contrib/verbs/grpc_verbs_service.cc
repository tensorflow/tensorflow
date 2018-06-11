/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_VERBS

#include "grpcpp/alarm.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_builder.h"

#include "tensorflow/contrib/verbs/grpc_verbs_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"

namespace tensorflow {

GrpcVerbsService::GrpcVerbsService(const WorkerEnv* worker_env,
                                   ::grpc::ServerBuilder* builder)
    : is_shutdown_(false), worker_env_(worker_env) {
  builder->RegisterService(&verbs_service_);
  cq_ = builder->AddCompletionQueue().release();
}

GrpcVerbsService::~GrpcVerbsService() {
  delete shutdown_alarm_;
  delete cq_;
}

void GrpcVerbsService::Shutdown() {
  bool did_shutdown = false;
  {
    mutex_lock l(shutdown_mu_);
    if (!is_shutdown_) {
      LOG(INFO) << "Shutting down GrpcWorkerService.";
      is_shutdown_ = true;
      did_shutdown = true;
    }
  }
  if (did_shutdown) {
    shutdown_alarm_ =
        new ::grpc::Alarm(cq_, gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
  }
}

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(GetRemoteAddress, false);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                             \
  do {                                                                       \
    mutex_lock l(shutdown_mu_);                                              \
    if (!is_shutdown_) {                                                     \
      Call<GrpcVerbsService, grpc::VerbsService::AsyncService,               \
           method##Request, method##Response>::                              \
          EnqueueRequest(&verbs_service_, cq_,                               \
                         &grpc::VerbsService::AsyncService::Request##method, \
                         &GrpcVerbsService::method##Handler,                 \
                         (supports_cancel));                                 \
    }                                                                        \
  } while (0)

// This method blocks forever handling requests from the completion queue.
void GrpcVerbsService::HandleRPCsLoop() {
  for (int i = 0; i < 10; ++i) {
    ENQUEUE_REQUEST(GetRemoteAddress, false);
  }

  void* tag;
  bool ok;

  while (cq_->Next(&tag, &ok)) {
    UntypedCall<GrpcVerbsService>::Tag* callback_tag =
        static_cast<UntypedCall<GrpcVerbsService>::Tag*>(tag);
    if (callback_tag) {
      callback_tag->OnCompleted(this, ok);
    } else {
      cq_->Shutdown();
    }
  }
}

void GrpcVerbsService::GetRemoteAddressHandler(
    WorkerCall<GetRemoteAddressRequest, GetRemoteAddressResponse>* call) {
  Status s = GetRemoteAddressSync(&call->request, &call->response);
  call->SendResponse(ToGrpcStatus(s));
  ENQUEUE_REQUEST(GetRemoteAddress, false);
}

// synchronous method
Status GrpcVerbsService::GetRemoteAddressSync(
    const GetRemoteAddressRequest* request,
    GetRemoteAddressResponse* response) {
  // analyzing request
  // the channel setting part is redundant.
  const string remote_host_name = request->host_name();
  RdmaChannel* rc = rdma_mgr_->FindChannel(remote_host_name);
  CHECK(rc);
  RdmaAddress ra;
  ra.lid = request->channel().lid();
  ra.qpn = request->channel().qpn();
  ra.psn = request->channel().psn();
  ra.snp = request->channel().snp();
  ra.iid = request->channel().iid();
  rc->SetRemoteAddress(ra, false);
  rc->Connect();
  int i = 0;
  int idx[] = {1, 0};
  std::vector<RdmaMessageBuffer*> mb(rc->message_buffers());
  CHECK_EQ(request->mr_size(), RdmaChannel::kNumMessageBuffers);
  for (const auto& mr : request->mr()) {
    // the connections are crossed, i.e.
    // local tx_message_buffer <---> remote rx_message_buffer_
    // local rx_message_buffer <---> remote tx_message_buffer_
    // hence idx[] = {1, 0}.
    RdmaMessageBuffer* rb = mb[idx[i]];
    RemoteMR rmr;
    rmr.remote_addr = mr.remote_addr();
    rmr.rkey = mr.rkey();
    rb->SetRemoteMR(rmr, false);
    i++;
  }
  CHECK(i == RdmaChannel::kNumMessageBuffers);

  // setting up response
  response->set_host_name(
      worker_env_->session_mgr->LegacySession()->worker_name);
  Channel* channel_info = response->mutable_channel();
  channel_info->set_lid(rc->self().lid);
  channel_info->set_qpn(rc->self().qpn);
  channel_info->set_psn(rc->self().psn);
  channel_info->set_snp(rc->self().snp);
  channel_info->set_iid(rc->self().iid);
  for (int i = 0; i < RdmaChannel::kNumMessageBuffers; i++) {
    MemoryRegion* mr = response->add_mr();
    mr->set_remote_addr(reinterpret_cast<uint64>(mb[i]->buffer()));
    mr->set_rkey(mb[i]->self()->rkey);
  }
  return Status::OK();
}

// Create a GrpcVerbsService, then assign it to a given handle.
void SetNewVerbsService(GrpcVerbsService** handle, const WorkerEnv* worker_env,
                        ::grpc::ServerBuilder* builder) {
  *handle = new GrpcVerbsService(worker_env, builder);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
