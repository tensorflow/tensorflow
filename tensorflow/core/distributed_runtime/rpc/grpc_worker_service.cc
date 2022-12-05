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
#include <memory>
#include <unordered_map>
#include <vector>

#include "grpcpp/alarm.h"
#include "grpcpp/server_builder.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_response_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/tsl/protobuf/rpc_options.pb.h"

namespace tensorflow {

namespace {

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
      tsl::Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,  \
                method##Request, method##Response>::                         \
          EnqueueRequestForMethod(                                           \
              worker_service_, cq_.get(),                                    \
              static_cast<int>(GrpcWorkerMethod::k##method),                 \
              &GrpcWorkerServiceThread::method##Handler, (supports_cancel)); \
    }                                                                        \
  } while (0)

#define SETUP_FOR_REQUEST(method, default_depth, supports_cancel)              \
  for (int i = 0;                                                              \
       i < gtl::FindWithDefault(queue_depth_,                                  \
                                static_cast<int>(GrpcWorkerMethod::k##method), \
                                default_depth);                                \
       ++i) {                                                                  \
    ENQUEUE_REQUEST(method, supports_cancel);                                  \
  }

// GrpcWorkerService spawns one or more GrpcWorkerServiceThreads to service
// requests.  Each thread operates on an independent completion queue.
class GrpcWorkerServiceThread {
 public:
  explicit GrpcWorkerServiceThread(
      GrpcWorker* worker, ::grpc::ServerBuilder* builder,
      std::unordered_map<int, int> queue_depth, GrpcResponseCache* cache,
      grpc::WorkerService::AsyncService* worker_service)
      : worker_(worker),
        queue_depth_(queue_depth),
        cache_(cache),
        worker_service_(worker_service),
        is_shutdown_(false) {
    cq_ = builder->AddCompletionQueue();
  }

  void Start() {
    thread_.reset(
        worker_->env()->env->StartThread(ThreadOptions(), "grpc_worker_service",
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
  // Add one or more completion queue entries for each worker method, then
  // begin servicing requests from the completion queue.
  void HandleRPCsLoop() {
    // TODO(ncteisen): This may require performance engineering. We can
    // change the number of threads, the number of handlers per thread,
    // or even decide to specialize certain threads to certain methods.
    SETUP_FOR_REQUEST(GetStatus, 1, false);
    SETUP_FOR_REQUEST(CreateWorkerSession, 1, false);
    SETUP_FOR_REQUEST(DeleteWorkerSession, 1, false);
    SETUP_FOR_REQUEST(CleanupAll, 1, false);
    SETUP_FOR_REQUEST(RegisterGraph, 1, false);
    SETUP_FOR_REQUEST(DeregisterGraph, 1, false);
    SETUP_FOR_REQUEST(Logging, 1, false);
    SETUP_FOR_REQUEST(Tracing, 1, false);
    SETUP_FOR_REQUEST(CompleteGroup, 10, true);
    SETUP_FOR_REQUEST(CompleteInstance, 10, true);
    SETUP_FOR_REQUEST(GetStepSequence, 10, true);
    SETUP_FOR_REQUEST(RecvBuf, 500, true);
    SETUP_FOR_REQUEST(RunGraph, 100, true);
    SETUP_FOR_REQUEST(CleanupGraph, 100, false);
    SETUP_FOR_REQUEST(MarkRecvFinished, 10, false);

    // TODO(ncteisen): Determine a better policy for enqueuing the
    // appropriate number of each request type.
    for (int i = 0;
         i < gtl::FindWithDefault(
                 queue_depth_, static_cast<int>(GrpcWorkerMethod::kRecvTensor),
                 1000);
         ++i) {
      EnqueueRecvTensorRequestRaw();
    }

    void* tag;
    bool ok;

    while (cq_->Next(&tag, &ok)) {
      tsl::UntypedCall<GrpcWorkerServiceThread>::Tag* callback_tag =
          static_cast<tsl::UntypedCall<GrpcWorkerServiceThread>::Tag*>(tag);
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
      tsl::Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,
                RequestMessage, ResponseMessage>;

  // Handle all non-cancellable simple methods with a standard wrapper.
  // The boolean `may_block_on_compute_pool` indicates whether or not the
  // operation may block on activities (such as op execution) that run on the
  // compute pool.
#define HANDLE_CALL(method, may_block_on_compute_pool)                        \
  void method##Handler(WorkerCall<method##Request, method##Response>* call) { \
    auto closure = [this, call]() {                                           \
      Status s = worker_->method(&call->request, &call->response);            \
      if (!s.ok()) {                                                          \
        VLOG(3) << "Bad response from " << #method << ": " << s;              \
      }                                                                       \
      call->SendResponse(ToGrpcStatus(s));                                    \
    };                                                                        \
    if ((may_block_on_compute_pool)) {                                        \
      worker_->env()->env->SchedClosure(std::move(closure));                  \
    } else {                                                                  \
      worker_->env()->compute_pool->Schedule(std::move(closure));             \
    }                                                                         \
    ENQUEUE_REQUEST(method, false);                                           \
  }

  HANDLE_CALL(GetStatus, false);
  HANDLE_CALL(CreateWorkerSession, false);
  HANDLE_CALL(DeleteWorkerSession, true);
  HANDLE_CALL(CleanupAll, false);
  HANDLE_CALL(RegisterGraph, false);
  HANDLE_CALL(DeregisterGraph, false);
  HANDLE_CALL(CleanupGraph, false);
  HANDLE_CALL(Logging, false);
  HANDLE_CALL(Tracing, false);

#undef HANDLE_CALL

  void GetStepSequenceHandler(
      WorkerCall<GetStepSequenceRequest, GetStepSequenceResponse>* call) {
    Schedule([this, call]() {
      worker_->GetStepSequenceAsync(
          &call->request, &call->response, [call](const Status& s) {
            VLOG(3) << "Bad response from GetStepSequence:" << s;
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    ENQUEUE_REQUEST(GetStepSequence, true);
  }

  void MarkRecvFinishedHandler(
      WorkerCall<MarkRecvFinishedRequest, MarkRecvFinishedResponse>* call) {
    VLOG(3) << "Clean cache entry for request " << call->request.request_id();
    worker_->RemoveCacheEntryForId(call->request.request_id());
    call->SendResponse(::grpc::Status::OK);
    ENQUEUE_REQUEST(MarkRecvFinished, false);
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
                               VLOG(3) << "RunGraph::Done";
                               if (!s.ok()) {
                                 VLOG(3) << "Bad response from RunGraph:" << s;
                               }
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

      worker_->GrpcRecvTensorAsync(
          call_opts, &call->request, &call->response,
          [call, call_opts](const Status& s) {
            call->ClearCancelCallback();
            delete call_opts;
            if (!s.ok()) {
              VLOG(3) << "Bad response from RecvTensor:" << s;
            }
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    EnqueueRecvTensorRequestRaw();
  }

  void RecvBufHandler(WorkerCall<RecvBufRequest, RecvBufResponse>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->RecvBufAsync(call_opts, &call->request, &call->response,
                            [call, call_opts](const Status& s) {
                              call->ClearCancelCallback();
                              delete call_opts;
                              if (!s.ok()) {
                                VLOG(3) << "Bad response from RecvBuf:" << s;
                              }
                              call->SendResponse(ToGrpcStatus(s));
                            });
    });
    ENQUEUE_REQUEST(RecvBuf, true);
  }

  void CompleteGroupHandler(
      WorkerCall<CompleteGroupRequest, CompleteGroupResponse>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->CompleteGroupAsync(
          call_opts, &call->request, &call->response,
          [call, call_opts](const Status& s) {
            call->ClearCancelCallback();
            delete call_opts;
            if (!s.ok()) {
              VLOG(3) << "Bad response from CompleteGroup:" << s;
            }
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    ENQUEUE_REQUEST(CompleteGroup, true);
  }

  void CompleteInstanceHandler(
      WorkerCall<CompleteInstanceRequest, CompleteInstanceResponse>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->CompleteInstanceAsync(
          call_opts, &call->request, &call->response,
          [call, call_opts](const Status& s) {
            call->ClearCancelCallback();
            delete call_opts;
            if (!s.ok()) {
              VLOG(3) << "Bad response from CompleteInstance:" << s;
            }
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    ENQUEUE_REQUEST(CompleteInstance, false);
  }
#undef ENQUEUE_REQUEST

  void EnqueueRecvTensorRequestRaw() {
    mutex_lock l(shutdown_mu_);
    if (!is_shutdown_) {
      tsl::Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,
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
  std::unordered_map<int, int> queue_depth_;
  GrpcResponseCache* cache_;
  grpc::WorkerService::AsyncService* const worker_service_;

  mutex shutdown_mu_;
  bool is_shutdown_ TF_GUARDED_BY(shutdown_mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerServiceThread);
};

class GrpcWorkerService : public tsl::AsyncServiceInterface {
 public:
  GrpcWorkerService(GrpcWorker* worker, ::grpc::ServerBuilder* builder,
                    GrpcWorkerServiceOptions options)
      : is_shutdown_(false) {
    builder->RegisterService(&worker_service_);

    for (int i = 0; i < options.num_serving_threads; i++) {
      threads_.emplace_back(
          new GrpcWorkerServiceThread(worker, builder, options.queue_depth,
                                      cache_.get(), &worker_service_));
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
  grpc::WorkerService::AsyncService worker_service_;
  std::vector<std::unique_ptr<GrpcWorkerServiceThread>> threads_;

  std::unique_ptr<GrpcResponseCache> cache_;
  mutex service_shutdown_mu_;
  bool is_shutdown_ TF_GUARDED_BY(service_shutdown_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerService);
};

}  // namespace

GrpcWorker::GrpcWorker(WorkerEnv* worker_env, const ConfigProto& config)
    : Worker(worker_env),
      recv_buf_max_chunk_(
          config.experimental().recv_buf_max_chunk() > 0
              ? config.experimental().recv_buf_max_chunk()
              : (config.experimental().recv_buf_max_chunk() < 0 ? 0 : 4096)) {
  if (config.rpc_options().cache_rpc_response()) {
    EnableResponseCache();
  }
}

void GrpcWorker::EnableResponseCache() {
  VLOG(3) << "Enabling gRPC tensor response cache.";
  response_cache_ = std::make_unique<GrpcResponseCache>();
}

// GrpcRecvTensorAsync: unlike the other Worker methods, which use protocol
// buffers for a response object, to avoid extra protocol buffer serialization
// overhead we generate our response directly into a ::grpc::ByteBuffer object
void GrpcWorker::GrpcRecvTensorAsync(CallOptions* opts,
                                     const RecvTensorRequest* request,
                                     ::grpc::ByteBuffer* response,
                                     StatusCallback done) {
  VLOG(3) << "GrpcRecvTensorAsync req: " << request->DebugString();
  const int64_t request_id = request->request_id();
  const int64_t step_id = request->step_id();

  bool cache_enabled = (response_cache_ != nullptr && request_id != 0);

  auto do_response = [response, done, cache_enabled](const Tensor& tensor,
                                                     bool is_dead,
                                                     const Status& status) {
    if (status.ok()) {
      grpc::EncodeTensorToByteBuffer(is_dead, tensor, cache_enabled, response);
    }
    done(status);
  };

  // If response cache is enabled and the response cache already contains the
  // request, we delegate this retry request to the response cache. Otherwise,
  // we add the request to the response cache and start the computation to
  // retrieve the requested data.
  if (cache_enabled &&
      response_cache_->QueueRequest(request_id, step_id, do_response)) {
    return;
  }

  auto rendezvous_done = [this, request_id, do_response, cache_enabled](
                             const Tensor& tensor, bool is_dead,
                             const Status& status) {
    if (cache_enabled) {
      // Data is ready. Process all pending requests in the response cache.
      response_cache_->OnRequestFinished(request_id, tensor, is_dead, status);
    } else {
      do_response(tensor, is_dead, status);
    }
  };

  auto fail = [&rendezvous_done](const Status& status) {
    rendezvous_done(Tensor(), false, status);
  };

  Status s = recent_request_ids_.TrackUnique(
      request_id, "RecvTensor (GrpcWorker)", *request);
  if (!s.ok()) {
    fail(s);
    return;
  }

  const string& key = request->rendezvous_key();
  TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
  Rendezvous::ParsedKey parsed;
  s = Rendezvous::ParseKey(key, &parsed);
  Device* src_dev = nullptr;
  if (s.ok()) {
    s = PrepareRecvTensor(parsed, &src_dev);
  }
  if (!s.ok()) {
    fail(s);
    return;
  }

  // Request the tensor associated with the rendezvous key.
  // Any time while waiting for the tensor to be produced, up until the start of
  // execution of the callback lambda body below, an RPC cancellation should
  // abort the rendezvous.
  // Note that gRPC can generate cancellations in response to transient network
  // failures, and the client might not observe any errors or cancellations but
  // simply waits for the responses. Aborting the step would report an error to
  // the client, and avoid permanent hanging in distributed function execution.
  opts->SetCancelCallback([this, step_id]() {
    LOG(WARNING) << "RecvTensor cancelled for " << step_id;
    AbortStep(step_id);
  });
  env_->rendezvous_mgr->RecvLocalAsync(
      step_id, parsed,
      [opts, rendezvous_done, src_dev, request](
          const Status& status, const Rendezvous::Args& send_args,
          const Rendezvous::Args& recv_args, const Tensor& val,
          const bool is_dead) {
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
            if (src_dev->tensorflow_accelerator_device_info() && (!on_host)) {
              DeviceContext* send_dev_context = send_args.device_context;
              AllocatorAttributes alloc_attrs;
              alloc_attrs.set_gpu_compatible(true);
              alloc_attrs.set_on_host(true);
              Allocator* alloc = src_dev->GetAllocator(alloc_attrs);
              Tensor* copy = new Tensor(alloc, val.dtype(), val.shape());
              CHECK(send_dev_context)
                  << "send dev name: " << src_dev->name() << " gpu_info: "
                  << src_dev->tensorflow_accelerator_device_info();
              // "val" is on an accelerator device. Uses the device_context to
              // fill the copy on host.
              StatusCallback copy_ready = [rendezvous_done, copy,
                                           is_dead](const Status& s) {
                // The value is now ready to be returned on the wire.
                rendezvous_done(*copy, is_dead, s);
                delete copy;
              };

              CopyDeviceToHost(&val, alloc, alloc, request->rendezvous_key(),
                               src_dev, copy, send_dev_context, copy_ready);
              return;
            }
          }
        }

        rendezvous_done(val, is_dead, status);
      });
}

namespace {
// If RecvBufRespExtra.tensor_content is a single large string, then gRPC
// can stall on the recv side when the string buffer needs to be enlarged,
// since the size is not sent in advance.  Changing this field to a sequence
// of small strings costs some extra time on the send side, since we do
// some otherwise unnecessary copies, but it improves runtime overall by
// improving flow control.  Best performance is likely achieved with a
// max_chunk_bytes equal to the memory page size.
//
// TODO(tucker): When proto3 supports [ctype=CORD] then change
// RecvBufRespExtra.tensor_content to a cord instead of a repeated string,
// and remove this function.
void SetTensorInRecvBufResp(int64_t max_chunk_bytes, const Tensor* tensor,
                            RecvBufResponse* response) {
  RecvBufRespExtra extra;
  int64_t num_bytes = tensor->TotalBytes();
  const char* head = reinterpret_cast<const char*>(DMAHelper::base(tensor));
  while (num_bytes > 0) {
    int64_t bytes =
        max_chunk_bytes > 0 ? std::min(num_bytes, max_chunk_bytes) : num_bytes;
    extra.add_tensor_content(std::string(head, bytes));
    head += bytes;
    num_bytes -= bytes;
  }
  response->mutable_transport_options()->PackFrom(extra);
}
}  // namespace

void GrpcWorker::RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                              RecvBufResponse* response, StatusCallback done) {
  const int64_t request_id = request->request_id();
  const int64_t step_id = request->step_id();
  bool cache_enabled = (response_cache_ != nullptr && request_id != 0);

  auto do_response = [this, response, done, cache_enabled](
                         const Tensor& tensor, bool is_dead,
                         const Status& status) {
    if (status.ok()) {
      SetTensorInRecvBufResp(recv_buf_max_chunk_, &tensor, response);
    }
    response->set_send_start_micros(env_->env->NowMicros());
    response->set_require_ack(cache_enabled);
    done(status);
  };

  // If response cache is enabled and the response cache already contains the
  // request, we delegate this retry request to the response cache. Otherwise,
  // we add the request to the response cache and start the computation to
  // retrieve the requested data.
  if (cache_enabled &&
      response_cache_->QueueRequest(request_id, step_id, do_response)) {
    return;
  }

  auto rendezvous_done = [this, request_id, do_response, cache_enabled](
                             const Tensor& tensor, const Status& status) {
    if (cache_enabled) {
      // Data is ready. Process all pending requests in the response cache.
      response_cache_->OnRequestFinished(request_id, tensor, false, status);
    } else {
      do_response(tensor, false, status);
    }
  };

  auto fail = [&rendezvous_done](const Status& status) {
    rendezvous_done(Tensor(), status);
  };

  // This is a generic, low performance implementation appropriate for grpc.
  Status s = recent_request_ids_.TrackUnique(request_id, "RecvBuf (GrpcWorker)",
                                             *request);
  if (!s.ok()) {
    fail(s);
    return;
  }

  CollectiveExecutor::Handle ce_handle(
      env_->collective_executor_mgr->FindOrCreate(step_id), true);
  CollectiveRemoteAccess* rma = ce_handle.get()->remote_access();
  auto consumer_callback = [this, request, rendezvous_done](
                               const Status& status,
                               BufRendezvous::Hook* hook) {
    Status s = status;
    if (s.ok()) {
      if (hook == nullptr) {
        s = errors::Internal("Invalid null hook for key ",
                             request->buf_rendezvous_key());
      }
      if (!DMAHelper::CanUseDMA(hook->prod_value)) {
        s = errors::Internal("Tensor value for key ",
                             request->buf_rendezvous_key(),
                             " is not of a type supported by RecvBuf");
      }
    } else {
      if (hook != nullptr) {
        LOG(ERROR) << "Got hook " << hook << " with status " << s
                   << " from ConsumeBuf";
      }
    }

    if (s.ok()) {
      // The RPC source tensor needs to be in CPU RAM.  If not already
      // there make a copy using memory appropriate to the purpose.
      const size_t num_bytes = hook->prod_value->TotalBytes();
      const bool on_host =
          hook->prod_dev->attributes().device_type() == "CPU" ||
          hook->prod_attr.on_host();
      if ((!on_host) && (num_bytes > 0)) {
        Device* cpu_dev = nullptr;
        s = env_->device_mgr->LookupDevice("CPU:0", &cpu_dev);
        if (s.ok()) {
          AllocatorAttributes cpu_attr;
          cpu_attr.set_gpu_compatible(true);
          cpu_attr.set_nic_compatible(true);
          profiler::ScopedMemoryDebugAnnotation op_annotation(
              "GrpcWorker::RecvBufAsync::consumer_callback", request->step_id(),
              "dynamic", hook->prod_value->dtype(),
              [hook]() { return hook->prod_value->shape().DebugString(); });
          Tensor* cpu_tensor =
              new Tensor(cpu_dev->GetAllocator(cpu_attr),
                         hook->prod_value->dtype(), hook->prod_value->shape());
          hook->prod_ctx->CopyDeviceTensorToCPU(
              hook->prod_value, "empty_name", hook->prod_dev, cpu_tensor,
              [hook, cpu_tensor, rendezvous_done](const Status& s) {
                rendezvous_done(*cpu_tensor, s);
                BufRendezvous::DoneWithHook(hook);
                delete cpu_tensor;
              });
          return;
        }
      }
    }

    if (hook == nullptr) {
      rendezvous_done(Tensor(), s);
    } else {
      rendezvous_done(*hook->prod_value, s);
      BufRendezvous::DoneWithHook(hook);
    }
  };
  rma->buf_rendezvous()->ConsumeBuf(
      request->buf_rendezvous_key(), request->src_device(),
      request->src_incarnation(), consumer_callback,
      /*cancellation_manager=*/nullptr);
}

void GrpcWorker::LoggingAsync(const LoggingRequest* request,
                              LoggingResponse* response, StatusCallback done) {
  auto env = this->env();
  if (env) {
    auto session_mgr = env->session_mgr;
    if (session_mgr) {
      if (request->enable_rpc_logging()) {
        session_mgr->SetLogging(true);
      }
      // NOTE(mrry): Handle old masters that disable RPC logging by setting
      // `request->enable_rpc_logging` to `false`.
      if (request->disable_rpc_logging() ||
          (!request->enable_rpc_logging() &&
           request->fetch_step_id_size() == 0)) {
        session_mgr->SetLogging(false);
      }
      for (const auto& step_id : request->fetch_step_id()) {
        session_mgr->RetrieveLogs(step_id, response);
      }
      if (request->clear()) {
        session_mgr->ClearLogs();
      }
    }
  }
  done(OkStatus());
}

void GrpcWorker::CleanupGraphAsync(const CleanupGraphRequest* request,
                                   CleanupGraphResponse* response,
                                   StatusCallback done) {
  if (response_cache_) {
    // Cleanup any stale response cache entries for this step. This can occur if
    // a worker crashes before acking a request.
    response_cache_->CleanEntriesForStep(request->step_id());
  }
  Worker::CleanupGraphAsync(request, response, done);
}

WorkerEnv* GrpcWorker::env() { return env_; }

void GrpcWorker::RemoveCacheEntryForId(int64_t request_id) {
  if (response_cache_) {
    response_cache_->EraseRequestId(request_id);
  }
}

std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* env,
                                          const ConfigProto& config) {
  return std::unique_ptr<GrpcWorker>(new GrpcWorker(env, config));
}

std::unique_ptr<tsl::AsyncServiceInterface> NewGrpcWorkerService(
    GrpcWorker* worker, ::grpc::ServerBuilder* builder,
    GrpcWorkerServiceOptions options) {
  return std::unique_ptr<tsl::AsyncServiceInterface>(
      new GrpcWorkerService(worker, builder, options));
}

}  // namespace tensorflow
