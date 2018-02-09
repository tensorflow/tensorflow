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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"

#include <deque>

#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif  // GOOGLE_CUDA
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {

class GrpcWorkerService : public AsyncServiceInterface {
  // TODO(ncteisen): consider adding a config var or flag for this
  static constexpr const size_t kGrpcWorkerServiceThreadCount = 8;

 public:
  GrpcWorkerService(GrpcWorker* worker, ::grpc::ServerBuilder* builder)
      : is_shutdown_(false) {
    builder->RegisterService(&worker_service_);
    for (int i = 0; i < kGrpcWorkerServiceThreadCount; i++) {
      threads_.emplace_back(
          new GrpcWorkerServiceThread(worker, builder, &worker_service_));
    }
  }

  void Shutdown() override {
    bool did_shutdown = false;
    {
      mutex_lock l(service_shutdown_mu_);
      if (!is_shutdown_) {
        LOG(INFO) << "Shutting down GrpcWorkerService.";
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      for (auto& worker_thread : threads_) {
        worker_thread->Shutdown();
      }
    }
  }

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(GetStatus, false);`), and enqueues it on
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
      Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,       \
           method##Request, method##Response>::                              \
          EnqueueRequestForMethod(                                           \
              worker_service_, cq_.get(),                                    \
              static_cast<int>(GrpcWorkerMethod::k##method),                 \
              &GrpcWorkerServiceThread::method##Handler, (supports_cancel)); \
    }                                                                        \
  } while (0)

  // This method blocks forever handling requests from the completion queue.
  void HandleRPCsLoop() override {
    for (auto& worker_thread : threads_) {
      worker_thread->Start();
    }
    for (auto& worker_thread : threads_) {
      worker_thread->Join();
    }
  }

 private:
  // Thread wrapping class that drives work over a single gRPC
  // CompletionQueue.
  class GrpcWorkerServiceThread {
   public:
    explicit GrpcWorkerServiceThread(
        GrpcWorker* worker, ::grpc::ServerBuilder* builder,
        grpc::WorkerService::AsyncService* worker_service)
        : worker_(worker),
          worker_service_(worker_service),
          is_shutdown_(false) {
      cq_ = builder->AddCompletionQueue();
    }

    void Start() {
      thread_.reset(worker_->env()->env->StartThread(
          ThreadOptions(), "grpc_worker_service",
          [this]() { HandleRPCsLoop(); }));
    }

    void Join() { thread_.reset(); }  // Blocks until thread exits

    void Shutdown() {
      {
        mutex_lock lock(shutdown_mu_);
        is_shutdown_ = true;
      }
      cq_->Shutdown();
    }

   private:
    void HandleRPCsLoop() {
      // TODO(ncteisen): This may require performance engineering. We can
      // change the number of threads, the number of handlers per thread,
      // or even decide to specialize certain threads to certain methods.
      ENQUEUE_REQUEST(GetStatus, false);
      ENQUEUE_REQUEST(CreateWorkerSession, false);
      ENQUEUE_REQUEST(DeleteWorkerSession, false);
      ENQUEUE_REQUEST(CleanupAll, false);
      ENQUEUE_REQUEST(RegisterGraph, false);
      ENQUEUE_REQUEST(DeregisterGraph, false);

      // TODO(ncteisen): Determine a better policy for enqueuing the
      // appropriate number of each request type.
      for (int i = 0; i < 1000; ++i) {
        EnqueueRecvTensorRequestRaw();
      }
      for (int i = 0; i < 100; ++i) {
        ENQUEUE_REQUEST(RunGraph, true);
      }
      for (int i = 0; i < 100; ++i) {
        ENQUEUE_REQUEST(CleanupGraph, false);
      }

      ENQUEUE_REQUEST(Logging, false);
      ENQUEUE_REQUEST(Tracing, false);

      void* tag;
      bool ok;

      while (cq_->Next(&tag, &ok)) {
        UntypedCall<GrpcWorkerServiceThread>::Tag* callback_tag =
            static_cast<UntypedCall<GrpcWorkerServiceThread>::Tag*>(tag);
        CHECK(callback_tag);
        callback_tag->OnCompleted(this, ok);
      }
    }

   private:
    void Schedule(std::function<void()> f) {
      worker_->env()->compute_pool->Schedule(std::move(f));
    }

    // The following section contains one request handler method per
    // RPC. The `FooHandler` method is called (indirectly) by
    // `HandleRPCsLoop()` when the next Foo RPC is received. Each
    // `FooHandler` call schedules a closure on `worker_->env()->compute_pool`,
    // and is responsible for requesting the next Foo call by calling
    // `ENQUEUE_REQUEST(Foo)`.

    template <class RequestMessage, class ResponseMessage>
    using WorkerCall =
        Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,
             RequestMessage, ResponseMessage>;

    void GetStatusHandler(
        WorkerCall<GetStatusRequest, GetStatusResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->GetStatus(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(GetStatus, false);
    }

    void CreateWorkerSessionHandler(
        WorkerCall<CreateWorkerSessionRequest, CreateWorkerSessionResponse>*
            call) {
      Schedule([this, call]() {
        Status s =
            worker_->CreateWorkerSession(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(CreateWorkerSession, false);
    }

    void DeleteWorkerSessionHandler(
        WorkerCall<DeleteWorkerSessionRequest, DeleteWorkerSessionResponse>*
            call) {
      Schedule([this, call]() {
        Status s =
            worker_->DeleteWorkerSession(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(DeleteWorkerSession, false);
    }

    void CleanupAllHandler(
        WorkerCall<CleanupAllRequest, CleanupAllResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->CleanupAll(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(CleanupAll, false);
    }

    void RegisterGraphHandler(
        WorkerCall<RegisterGraphRequest, RegisterGraphResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->RegisterGraph(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(RegisterGraph, false);
    }

    void DeregisterGraphHandler(
        WorkerCall<DeregisterGraphRequest, DeregisterGraphResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->DeregisterGraph(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(DeregisterGraph, false);
    }

    void RunGraphHandler(WorkerCall<RunGraphRequest, RunGraphResponse>* call) {
      Schedule([this, call]() {
        CallOptions* call_opts = new CallOptions;
        ProtoRunGraphRequest* wrapped_request =
            new ProtoRunGraphRequest(&call->request);
        NonOwnedProtoRunGraphResponse* wrapped_response =
            new NonOwnedProtoRunGraphResponse(&call->response);
        call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
        worker_->RunGraphAsync(call_opts, wrapped_request, wrapped_response,
                               [call, call_opts, wrapped_request,
                                wrapped_response](const Status& s) {
                                 call->ClearCancelCallback();
                                 delete call_opts;
                                 delete wrapped_request;
                                 delete wrapped_response;
                                 call->SendResponse(ToGrpcStatus(s));
                               });
      });
      ENQUEUE_REQUEST(RunGraph, true);
    }

    void RecvTensorHandlerRaw(
        WorkerCall<RecvTensorRequest, ::grpc::ByteBuffer>* call) {
      Schedule([this, call]() {
        CallOptions* call_opts = new CallOptions;
        call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
        worker_->GrpcRecvTensorAsync(call_opts, &call->request, &call->response,
                                     [call, call_opts](const Status& s) {
                                       call->ClearCancelCallback();
                                       delete call_opts;
                                       call->SendResponse(ToGrpcStatus(s));
                                     });
      });
      EnqueueRecvTensorRequestRaw();
    }

    void CleanupGraphHandler(
        WorkerCall<CleanupGraphRequest, CleanupGraphResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->CleanupGraph(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(CleanupGraph, false);
    }

    void LoggingHandler(WorkerCall<LoggingRequest, LoggingResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->Logging(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(Logging, false);
    }

    void TracingHandler(WorkerCall<TracingRequest, TracingResponse>* call) {
      Schedule([this, call]() {
        Status s = worker_->Tracing(&call->request, &call->response);
        call->SendResponse(ToGrpcStatus(s));
      });
      ENQUEUE_REQUEST(Tracing, false);
    }
#undef ENQUEUE_REQUEST

    void EnqueueRecvTensorRequestRaw() {
      mutex_lock l(shutdown_mu_);
      if (!is_shutdown_) {
        Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,
             RecvTensorRequest, ::grpc::ByteBuffer>::
            EnqueueRequestForMethod(
                worker_service_, cq_.get(),
                static_cast<int>(GrpcWorkerMethod::kRecvTensor),
                &GrpcWorkerServiceThread::RecvTensorHandlerRaw,
                true /* supports cancel*/);
      }
    }

    GrpcWorker* const worker_ = nullptr;  // Not owned.
    std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
    std::unique_ptr<Thread> thread_;
    grpc::WorkerService::AsyncService* const worker_service_;

    mutex shutdown_mu_;
    bool is_shutdown_ GUARDED_BY(shutdown_mu_);
    TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerServiceThread);
  };  // GrpcWorkerServiceThread

  grpc::WorkerService::AsyncService worker_service_;
  std::vector<std::unique_ptr<GrpcWorkerServiceThread>> threads_;

  mutex service_shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(service_shutdown_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerService);
};

}  // namespace

GrpcWorker::GrpcWorker(WorkerEnv* worker_env)
    : Worker(worker_env), recv_tensor_recent_request_ids_(100000) {}

// GrpcRecvTensorAsync: unlike the other Worker methods, which use protocol
// buffers for a response object, to avoid extra protocol buffer serialization
// overhead we generate our response directly into a ::grpc::ByteBuffer object
void GrpcWorker::GrpcRecvTensorAsync(CallOptions* opts,
                                     const RecvTensorRequest* request,
                                     ::grpc::ByteBuffer* response,
                                     StatusCallback done) {
  Status s = recv_tensor_recent_request_ids_.TrackUnique(
      request->request_id(), "RecvTensor (GrpcWorker)", *request);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64 step_id = request->step_id();
  const string& key = request->rendezvous_key();
  TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
  Rendezvous::ParsedKey parsed;
  s = Rendezvous::ParseKey(key, &parsed);
  Device* src_dev = nullptr;
  if (s.ok()) {
    s = PrepareRecvTensor(parsed, &src_dev);
  }
  if (!s.ok()) {
    done(s);
    return;
  }

  // Request the tensor associated with the rendezvous key. Any time
  // while waiting for the tensor to be produced, up until the start
  // of execution of the callback lambda body below, an RPC
  // cancellation should abort the rendezvous.
  opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
  env_->rendezvous_mgr->RecvLocalAsync(
      step_id, parsed,
      [opts, response, done, src_dev](const Status& status,
                                      const Rendezvous::Args& send_args,
                                      const Rendezvous::Args& recv_args,
                                      const Tensor& val, const bool is_dead) {
        opts->ClearCancelCallback();
        if (status.ok()) {
          // DMA can only be used for Tensors that do not fall into
          // the following three odd edge cases: 1) a zero-size
          // buffer, 2) a dead tensor which has an uninit value, and
          // 3) the tensor has the on_host allocation attribute,
          // i.e. it's in CPU RAM *independent of its assigned
          // device type*.
          const bool on_host = send_args.alloc_attrs.on_host();
          {
            // Non-DMA cases.
            if (src_dev->tensorflow_gpu_device_info() && (!on_host)) {
#if GOOGLE_CUDA
              const DeviceContext* send_dev_context = send_args.device_context;
              AllocatorAttributes alloc_attrs;
              alloc_attrs.set_gpu_compatible(true);
              alloc_attrs.set_on_host(true);
              Allocator* alloc = src_dev->GetAllocator(alloc_attrs);
              Tensor* copy = new Tensor(alloc, val.dtype(), val.shape());
              CHECK(send_dev_context)
                  << "send dev name: " << src_dev->name()
                  << " gpu_info: " << src_dev->tensorflow_gpu_device_info();
              // "val" is on a GPU. Uses GPUUtil to fill the copy on host.
              StatusCallback copy_ready = [response, done, copy,
                                           is_dead](const Status& s) {
                // The value is now ready to be returned on the wire.
                grpc::EncodeTensorToByteBuffer(is_dead, *copy, response);
                done(s);
                delete copy;
              };

              GPUUtil::CopyGPUTensorToCPU(src_dev, send_dev_context, &val, copy,
                                          copy_ready);
#else
              done(errors::Internal("No GPU device in process"));
#endif  // GOOGLE_CUDA
            } else {
              grpc::EncodeTensorToByteBuffer(is_dead, val, response);
              done(Status::OK());
            }
          }
        } else {
          //  !s.ok()
          done(status);
        }
      });
}

void GrpcWorker::LoggingAsync(const LoggingRequest* request,
                              LoggingResponse* response, StatusCallback done) {
  auto env = this->env();
  if (env) {
    auto session_mgr = (SessionMgr*)env->session_mgr;
    if (session_mgr) {
      session_mgr->SetLogging(request->rpc_logging());
      for (const auto& step_id : request->fetch_step_id()) {
        session_mgr->RetrieveLogs(step_id, response);
      }
      if (request->clear()) {
        session_mgr->ClearLogs();
      }
    }
  }
  done(Status::OK());
}

WorkerEnv* GrpcWorker::env() { return env_; }

std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* env) {
  return std::unique_ptr<GrpcWorker>(new GrpcWorker(env));
}

std::unique_ptr<AsyncServiceInterface> NewGrpcWorkerService(
    GrpcWorker* worker, ::grpc::ServerBuilder* builder) {
  return std::unique_ptr<AsyncServiceInterface>(
      new GrpcWorkerService(worker, builder));
}

}  // namespace tensorflow
