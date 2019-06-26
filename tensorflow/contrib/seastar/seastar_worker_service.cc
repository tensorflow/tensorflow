#include "tensorflow/contrib/seastar/seastar_worker_service.h"

#include "tensorflow/contrib/seastar/seastar_tag_factory.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

template <class RequestMessage, class ResponseMessage>

class SeastarCall {
 public:
  RequestMessage req_;
  ResponseMessage resp_;
};

}  // namespace

using HandleRequestFunction = void (SeastarWorkerService::*)(SeastarServerTag*);

SeastarWorkerService::SeastarWorkerService(SeastarWorker* worker)
    : worker_(worker) {
  handler_map_[SeastarWorkerServiceMethod::kRunGraph] =
      &SeastarWorkerService::RunGraphHandler;
  handler_map_[SeastarWorkerServiceMethod::kRecvTensor] =
      &SeastarWorkerService::RecvTensorHandlerRaw;
  handler_map_[SeastarWorkerServiceMethod::kGetStatus] =
      &SeastarWorkerService::GetStatusHandler;
  handler_map_[SeastarWorkerServiceMethod::kCreateWorkerSession] =
      &SeastarWorkerService::CreateWorkerSessionHandler;
  handler_map_[SeastarWorkerServiceMethod::kDeleteWorkerSession] =
      &SeastarWorkerService::DeleteWorkerSessionHandler;
  handler_map_[SeastarWorkerServiceMethod::kRegisterGraph] =
      &SeastarWorkerService::RegisterGraphHandler;
  handler_map_[SeastarWorkerServiceMethod::kDeregisterGraph] =
      &SeastarWorkerService::DeregisterGraphHandler;
  handler_map_[SeastarWorkerServiceMethod::kCleanupGraph] =
      &SeastarWorkerService::CleanupGraphHandler;
  handler_map_[SeastarWorkerServiceMethod::kCleanupAll] =
      &SeastarWorkerService::CleanupAllHandler;
  handler_map_[SeastarWorkerServiceMethod::kLogging] =
      &SeastarWorkerService::LoggingHandler;
  handler_map_[SeastarWorkerServiceMethod::kTracing] =
      &SeastarWorkerService::TracingHandler;
  handler_map_[SeastarWorkerServiceMethod::kRecvBuf] =
      &SeastarWorkerService::RecvBufHandler;
  handler_map_[SeastarWorkerServiceMethod::kCompleteGroup] =
      &SeastarWorkerService::CompleteGroupHandler;
  handler_map_[SeastarWorkerServiceMethod::kCompleteInstance] =
      &SeastarWorkerService::CompleteInstanceHandler;
  handler_map_[SeastarWorkerServiceMethod::kGetStepSequence] =
      &SeastarWorkerService::GetStepSequenceHandler;
}

HandleRequestFunction SeastarWorkerService::GetHandler(
    SeastarWorkerServiceMethod methodId) {
  return handler_map_[methodId];
}

void SeastarWorkerService::RunGraphHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<RunGraphRequest, RunGraphResponse>* call =
        new SeastarCall<RunGraphRequest, RunGraphResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    CallOptions* call_opts = new CallOptions;
    ProtoRunGraphRequest* wrapped_request =
        new ProtoRunGraphRequest(&call->req_);
    NonOwnedProtoRunGraphResponse* wrapped_response =
        new NonOwnedProtoRunGraphResponse(&call->resp_);
    worker_->RunGraphAsync(call_opts, wrapped_request, wrapped_response,
                           [tag, call, call_opts, wrapped_request,
                            wrapped_response](const Status& s) {
                             tag->ProcessDone(s);
                             delete call_opts;
                             delete wrapped_request;
                             delete wrapped_response;
                             delete call;
                           });
  });
}

void SeastarWorkerService::GetStatusHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<GetStatusRequest, GetStatusResponse>* call =
        new SeastarCall<GetStatusRequest, GetStatusResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->GetStatus(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::CreateWorkerSessionHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<CreateWorkerSessionRequest, CreateWorkerSessionResponse>* call =
        new SeastarCall<CreateWorkerSessionRequest,
                        CreateWorkerSessionResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->CreateWorkerSession(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::DeleteWorkerSessionHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<DeleteWorkerSessionRequest, DeleteWorkerSessionResponse>* call =
        new SeastarCall<DeleteWorkerSessionRequest,
                        DeleteWorkerSessionResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->DeleteWorkerSession(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::CleanupAllHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<CleanupAllRequest, CleanupAllResponse>* call =
        new SeastarCall<CleanupAllRequest, CleanupAllResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->CleanupAll(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::RegisterGraphHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<RegisterGraphRequest, RegisterGraphResponse>* call =
        new SeastarCall<RegisterGraphRequest, RegisterGraphResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->RegisterGraph(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::DeregisterGraphHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<DeregisterGraphRequest, DeregisterGraphResponse>* call =
        new SeastarCall<DeregisterGraphRequest, DeregisterGraphResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->DeregisterGraph(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::CleanupGraphHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<CleanupGraphRequest, CleanupGraphResponse>* call =
        new SeastarCall<CleanupGraphRequest, CleanupGraphResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);
    Status s = worker_->CleanupGraph(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::LoggingHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<LoggingRequest, LoggingResponse>* call =
        new SeastarCall<LoggingRequest, LoggingResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);

    Status s = worker_->Logging(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::TracingHandler(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    SeastarCall<TracingRequest, TracingResponse>* call =
        new SeastarCall<TracingRequest, TracingResponse>();
    InitSeastarServerTag(&call->req_, &call->resp_, tag);

    Status s = worker_->Tracing(&call->req_, &call->resp_);
    tag->ProcessDone(s);
    delete call;
  });
}

void SeastarWorkerService::RecvBufHandler(SeastarServerTag* tag) {
  tag->ProcessDone(
      errors::Unimplemented("SeastarWorkerService::RecvBufHandler()"));
}

void SeastarWorkerService::CompleteGroupHandler(SeastarServerTag* tag) {
  tag->ProcessDone(
      errors::Unimplemented("SeastarWorkerService::CompleteGroupHandler()"));
}

void SeastarWorkerService::CompleteInstanceHandler(SeastarServerTag* tag) {
  tag->ProcessDone(
      errors::Unimplemented("SeastarWorkerService::CompleteInstanceHandler()"));
}

void SeastarWorkerService::GetStepSequenceHandler(SeastarServerTag* tag) {
  tag->ProcessDone(
      errors::Unimplemented("SeastarWorkerService::GetStepSequenceHandler()"));
}

void SeastarWorkerService::RecvTensorHandlerRaw(SeastarServerTag* tag) {
  Schedule([this, tag]() {
    CallOptions* call_opts = new CallOptions;

    SeastarCall<RecvTensorRequest, SeastarTensorResponse>* call =
        new SeastarCall<RecvTensorRequest, SeastarTensorResponse>();

    InitSeastarServerTag(&call->req_, &call->resp_, tag,
                         [call](const Status& s) { delete call; });

    worker_->RecvTensorAsync(call_opts, &call->req_, &call->resp_,
                             [tag, call, call_opts](const Status& s) {
                               delete call_opts;
                               tag->ProcessDone(s);
                             });
  });
}

void SeastarWorkerService::Schedule(std::function<void()> f) {
  worker_->env()->compute_pool->Schedule(std::move(f));
}

WorkerEnv* SeastarWorker::env() { return env_; }

SeastarWorker::SeastarWorker(WorkerEnv* worker_env) : Worker(worker_env) {}

void SeastarWorker::RecvTensorAsync(CallOptions* opts,
                                    const RecvTensorRequest* request,
                                    SeastarTensorResponse* response,
                                    StatusCallback done) {
  const int64 step_id = request->step_id();
  const string& key = request->rendezvous_key();
  Rendezvous::ParsedKey parsed;

  Status s = Rendezvous::ParseKey(key, &parsed);
  Device* src_dev = nullptr;
  if (s.ok()) {
    s = PrepareRecvTensor(parsed, &src_dev);
  }
  if (!s.ok()) {
    LOG(WARNING) << "PrepareRecvTensor failed, tensor:" << key;
    done(s);
    abort();
  }

  // opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
  env_->rendezvous_mgr->RecvLocalAsync(
      step_id, parsed,
      [opts, request, response, done, src_dev, key](
          const Status& status, const Rendezvous::Args& send_args,
          const Rendezvous::Args& recv_args, const Tensor& val,
          const bool is_dead) {
        // opts->ClearCancelCallback();

        if (!status.ok()) {
          LOG(WARNING)
              << "env_->rendezvous_mgr->RecvLocalAsync failed, error msg is: "
              << status.error_message();
        }

        if (status.ok()) {
          response->SetIsDead(is_dead);
          bool can_memcpy = DataTypeCanUseMemcpy(val.dtype());

          if (src_dev->tensorflow_gpu_device_info() &&
              (!send_args.alloc_attrs.on_host())) {
            CHECK(send_args.device_context)
                << "send dev name: " << src_dev->name()
                << " gpu_info: " << src_dev->tensorflow_gpu_device_info();

            AllocatorAttributes alloc_attrs;
            alloc_attrs.set_gpu_compatible(true);
            alloc_attrs.set_on_host(true);
            Allocator* alloc = src_dev->GetAllocator(alloc_attrs);
            Tensor* cpu_copy = new Tensor(alloc, val.dtype(), val.shape());

            send_args.device_context->CopyDeviceTensorToCPU(
                &val, request->rendezvous_key(), src_dev, cpu_copy,
                [response, cpu_copy, done](const Status& s) {
                  CHECK(s.ok()) << "copy tensor from gpu sync";
                  response->SetTensor(*cpu_copy);
                  delete cpu_copy;
                  done(s);
                });
          } else {
            // tensor is in CPU memory.
            response->SetTensor(val);
            if (!can_memcpy) {
              val.AsProtoTensorContent(&response->GetTensorProto());
            }
            done(Status());
          }
        } else {
          // !s.ok()
          done(status);
        }
      });
}

std::unique_ptr<SeastarWorker> NewSeastarWorker(WorkerEnv* worker_env) {
  return std::unique_ptr<SeastarWorker>(new SeastarWorker(worker_env));
}

std::unique_ptr<SeastarWorkerService> NewSeastarWorkerService(
    SeastarWorker* worker) {
  return std::unique_ptr<SeastarWorkerService>(
      new SeastarWorkerService(worker));
}

}  // namespace tensorflow
