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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"

#include <deque>

#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/protobuf/worker_service.grpc.pb.h"
#include "tensorflow/core/protobuf/worker_service.pb.h"

namespace tensorflow {

namespace {

static Tensor empty_tensor(DT_FLOAT);

class GrpcWorkerService : public AsyncServiceInterface {
 public:
  GrpcWorkerService(WorkerEnv* env, ::grpc::ServerBuilder* builder)
      : env_(env),
        cancellation_manager_(new CancellationManager),
        is_shutdown_(false) {
    builder->RegisterService(&worker_service_);
    cq_ = builder->AddCompletionQueue().release();
  }

  ~GrpcWorkerService() {
    delete shutdown_alarm_;
    delete cq_;
    delete cancellation_manager_;
  }

  void Shutdown() override {
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
      // NOTE(mrry): This enqueues a special event (with a null tag)
      // that causes the completion queue to be shut down on the
      // polling thread.
      shutdown_alarm_ =
          new ::grpc::Alarm(cq_, gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(GetStatus);`), and enqueues it on
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
    mutex_lock l(shutdown_mu_);                                               \
    if (!is_shutdown_) {                                                      \
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,              \
           method##Request, method##Response>::                               \
          EnqueueRequest(&worker_service_, cq_,                               \
                         &grpc::WorkerService::AsyncService::Request##method, \
                         &GrpcWorkerService::method##Handler,                 \
                         (supports_cancel));                                  \
    }                                                                         \
  } while (0)

  // This method blocks forever handling requests from the completion queue.
  void HandleRPCsLoop() {
    // TODO(mrry): This may require performance engineering. We can
    // add more threads to service the completion queue, and add more
    // of various request types if they are short and frequent.
    // Currently we allow unbounded numbers of pending calls for each
    // method, by re-enqueuing a request before the previous one
    // completes, and we may decide to bound some of the request
    // types.
    ENQUEUE_REQUEST(GetStatus, false);
    ENQUEUE_REQUEST(CleanupAll, false);
    ENQUEUE_REQUEST(RegisterGraph, false);
    ENQUEUE_REQUEST(DeregisterGraph, false);

    // TODO(mrry): Determine a better policy for enqueuing the appropriate
    // number of each request type.
    for (int i = 0; i < 1000; ++i) {
      ENQUEUE_REQUEST(RecvTensor, true);
    }
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(RunGraph, true);
    }

    ENQUEUE_REQUEST(CleanupGraph, false);
    ENQUEUE_REQUEST(Logging, false);
    ENQUEUE_REQUEST(Tracing, false);

    void* tag;
    bool ok;

    while (cq_->Next(&tag, &ok)) {
      UntypedCall<GrpcWorkerService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
      if (callback_tag) {
        callback_tag->OnCompleted(this, ok);
        delete callback_tag;
      } else {
        // NOTE(mrry): A null `callback_tag` indicates that this is
        // the shutdown alarm.
        cq_->Shutdown();
      }
    }
  }

 private:
  WorkerEnv* env_;                     // Not owned.
  ::grpc::ServerCompletionQueue* cq_;  // Owned.

  grpc::WorkerService::AsyncService worker_service_;

  mutex mu_;
  CancellationManager* cancellation_manager_ GUARDED_BY(mu_);

  mutex shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(shutdown_mu_);
  ::grpc::Alarm* shutdown_alarm_;

  // The following section contains one request handler method per
  // RPC. The `FooHandler` method is called (indirectly) by
  // `HandleRPCsLoop()` when the next Foo RPC is received. Each
  // `FooHandler` call schedules a closure on `env_->compute_pool`,
  // and is responsible for requesting the next Foo call by calling
  // `ENQUEUE_REQUEST(Foo)`.

  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void GetStatusHandler(WorkerCall<GetStatusRequest, GetStatusResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      DeviceMgr* dm = env_->device_mgr;
      std::vector<DeviceAttributes> devices;
      dm->ListDeviceAttributes(&devices);
      call->response.mutable_device_attributes()->Reserve(devices.size());
      for (size_t i = 0; i < devices.size(); i++) {
        call->response.add_device_attributes()->Swap(&devices[i]);
      }
      call->SendResponse(::grpc::Status::OK);
    });
    ENQUEUE_REQUEST(GetStatus, false);
  }

  void CleanupAllHandler(
      WorkerCall<CleanupAllRequest, CleanupAllResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      std::vector<string> containers;
      for (const auto& c : call->request.container()) containers.push_back(c);
      env_->device_mgr->ClearContainers(containers);
      call->SendResponse(::grpc::Status::OK);
    });
    ENQUEUE_REQUEST(CleanupAll, false);
  }

  void RegisterGraphHandler(
      WorkerCall<RegisterGraphRequest, RegisterGraphResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      Status s = env_->graph_mgr->Register(
          call->request.session_handle(), call->request.graph_def(),
          call->request.graph_options(), call->response.mutable_graph_handle());
      call->SendResponse(ToGrpcStatus(s));
    });
    ENQUEUE_REQUEST(RegisterGraph, false);
  }

  void DeregisterGraphHandler(
      WorkerCall<DeregisterGraphRequest, DeregisterGraphResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      Status s = env_->graph_mgr->Deregister(call->request.graph_handle());
      call->SendResponse(ToGrpcStatus(s));
    });
    ENQUEUE_REQUEST(DeregisterGraph, false);
  }

  void RunGraphHandler(WorkerCall<RunGraphRequest, RunGraphResponse>* call) {
    env_->compute_pool->Schedule([this, call]() { DoRunGraph(call); });
    ENQUEUE_REQUEST(RunGraph, true);
  }

  void RecvTensorHandler(
      WorkerCall<RecvTensorRequest, RecvTensorResponse>* call) {
    env_->compute_pool->Schedule([this, call]() { DoRecvTensor(call); });
    ENQUEUE_REQUEST(RecvTensor, true);
  }

  void CleanupGraphHandler(
      WorkerCall<CleanupGraphRequest, CleanupGraphResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      const int64 step_id = call->request.step_id();
      env_->rendezvous_mgr->Cleanup(step_id);
      call->SendResponse(::grpc::Status::OK);
    });
    ENQUEUE_REQUEST(CleanupGraph, false);
  }

  void LoggingHandler(WorkerCall<LoggingRequest, LoggingResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      Status s = DoLogging(call);
      call->SendResponse(ToGrpcStatus(s));
    });
    ENQUEUE_REQUEST(Logging, false);
  }

  void TracingHandler(WorkerCall<TracingRequest, TracingResponse>* call) {
    SchedClosure([this, call]() {
      Status s = DoTracing(call);
      call->SendResponse(ToGrpcStatus(s));
    });
    ENQUEUE_REQUEST(Tracing, false);
  }
#undef ENQUEUE_REQUEST

 private:
  // The following section contains the implementation of RunGraph()
  // RecvTensor(), Logging(), and Tracing(), which are the four
  // non-trivial and potentially long-running RPCs performed by a
  // TensorFlow worker.

  void AbortStep(int64 step_id) {
    Rendezvous* rendez = env_->rendezvous_mgr->Find(step_id);
    SchedNonBlockingClosureAfter(1000000, [rendez, step_id]() {
      // Delay a bit before aborting the step. This way, the root
      // cause may return first back to the client instead of this
      // cancellation generated abort error.
      rendez->StartAbort(errors::Aborted("Step ", step_id));
      rendez->Unref();
    });
  }

  Status PrepareRunGraph(const RunGraphRequest& req, GraphMgr::NamedTensors* in,
                         GraphMgr::NamedTensors* out) {
    if (req.send_size() > 0) {
      // TODO(zhifengc): Let the caller decide on which device to
      // allocate the tensor.
      Device* cpu_dev = nullptr;
      TF_RETURN_IF_ERROR(env_->device_mgr->LookupDevice("CPU:0", &cpu_dev));
      AllocatorAttributes alloc_attrs;
      Tensor val;
      for (const NamedTensor& entry : req.send()) {
        TF_RETURN_IF_ERROR(
            cpu_dev->MakeTensorFromProto(entry.val(), alloc_attrs, &val));
        in->insert({entry.key(), val});
      }
    }
    for (const string& key : req.recv_key()) {
      out->insert({key, empty_tensor});
    }
    return Status::OK();
  }

  void DoRunGraph(WorkerCall<RunGraphRequest, RunGraphResponse>* call) {
    const int64 step_id = call->request.step_id();
    TRACEPRINTF("RunGraph: %lld", step_id);
    GraphMgr::NamedTensors in;
    GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
    Status s = PrepareRunGraph(call->request, &in, out);
    if (!s.ok()) {
      delete out;
      call->SendResponse(ToGrpcStatus(s));
      return;
    }
    StepStatsCollector* collector = nullptr;
    // TODO(mrry): Collect results from a profiler if available.
    CancellationManager* cm = new CancellationManager;
    call->SetCancelCallback([this, cm, step_id]() {
      cm->StartCancel();
      AbortStep(step_id);
    });
    CancellationToken token;
    {
      mutex_lock l(mu_);
      token = cancellation_manager_->get_cancellation_token();
      cancellation_manager_->RegisterCallback(token,
                                              [cm]() { cm->StartCancel(); });
    }
    env_->graph_mgr->ExecuteAsync(
        call->request.graph_handle(), step_id, call->request.exec_opts(),
        collector, cm, in, out, [this, call, cm, out, token](Status s) {
          call->ClearCancelCallback();
          {
            mutex_lock l(mu_);
            cancellation_manager_->DeregisterCallback(token);
          }
          delete cm;

          if (s.ok()) {
            for (const auto& p : *out) {
              const string& key = p.first;
              const Tensor& val = p.second;
              auto* recv = call->response.add_recv();
              recv->set_key(key);
              // TODO(zhifengc): Deal with gpu -> cpu copy.
              TensorProto* proto = recv->mutable_val();
              val.AsProtoField(proto);
            }
          }
          delete out;
          call->SendResponse(ToGrpcStatus(s));
        });
  }

  // Helper for RecvTensor. Validates "key" and returns the source
  // device in "*src_dev".
  Status PrepareRecvTensor(const string& key, Device** src_dev) {
    // Validate the key.
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, &parsed));

    // Figures out which device the tensor is hosted on.
    TF_RETURN_IF_ERROR(
        env_->device_mgr->LookupDevice(parsed.src_device, src_dev));

    // Does the device have the right incarnation number we expect?
    if ((*src_dev)->attributes().incarnation() != parsed.src_incarnation) {
      return errors::Aborted(
          "RecvTensor expects a different device incarnation: ",
          parsed.src_incarnation, " vs. ",
          (*src_dev)->attributes().incarnation(),
          ". Your worker job was probably restarted. Check your "
          "worker job for the reason why it was restarted.");
    }

    return Status::OK();
  }

  void DoRecvTensor(WorkerCall<RecvTensorRequest, RecvTensorResponse>* call) {
    const int64 step_id = call->request.step_id();
    const string& key = call->request.rendezvous_key();
    TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
    Device* src_dev = nullptr;
    Status s = PrepareRecvTensor(key, &src_dev);
    if (!s.ok()) {
      call->SendResponse(ToGrpcStatus(s));
      return;
    }

    // Request the tensor associated with the rendezvous key. Any time
    // while waiting for the tensor to be produced, up until the start
    // of execution of the callback lambda body below, an RPC
    // cancellation should abort the rendezvous.
    call->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
    env_->rendezvous_mgr->RecvLocalAsync(
        step_id, key,
        [this, call, src_dev](const Status& status,
                              const Rendezvous::Args& send_args,
                              const Rendezvous::Args& recv_args,
                              const Tensor& val, const bool is_dead) {
          call->ClearCancelCallback();
          Status s = status;
          if (s.ok()) {
            // DMA can only be used for Tensors that do not fall into
            // the following three odd edge cases: 1) a zero-size
            // buffer, 2) a dead tensor which has an uninit value, and
            // 3) the tensor has the on_host allocation attribute,
            // i.e. it's in CPU RAM *independent of its assigned
            // device type*.
            // const size_t bytes = is_dead ? 0 : val.TotalBytes();
            const bool on_host = send_args.alloc_attrs.on_host();
            const DeviceContext* send_dev_context = send_args.device_context;
            call->response.set_is_dead(is_dead);
            StatusCallback response_ready = [call](const Status& s) {
              // The value is now ready to be returned on the wire.
              call->response.set_send_start_micros(Env::Default()->NowMicros());
              call->SendResponse(ToGrpcStatus(s));
            };
            {
              // Non-DMA cases.
              if (src_dev->tensorflow_gpu_device_info() && (!on_host)) {
                CHECK(send_dev_context)
                    << "send dev name: " << src_dev->name()
                    << " gpu_info: " << src_dev->tensorflow_gpu_device_info();
                // "val" is on a GPU. Uses GPUUtil to fill the response proto.
                GPUUtil::SetProtoFromGPU(val, src_dev, send_dev_context,
                                         call->response.mutable_tensor(),
                                         is_dead, response_ready);
              } else {
                // "val" is in CPU memory.
                TensorProto* proto = call->response.mutable_tensor();
                val.AsProtoTensorContent(proto);
                response_ready(Status::OK());
              }
            }
          } else {
            //  !s.ok()
            call->SendResponse(ToGrpcStatus(s));
          }
        });
  }

  Status DoLogging(WorkerCall<LoggingRequest, LoggingResponse>* call) {
    // TODO(mrry): Platform-specific tracing support.
    return errors::Unimplemented("Logging");
  }

  Status DoTracing(WorkerCall<TracingRequest, TracingResponse>* call) {
    // TODO(mrry): Platform-specific tracing support.
    return errors::Unimplemented("Tracing");
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerService);
};

}  // namespace

AsyncServiceInterface* NewGrpcWorkerService(WorkerEnv* env,
                                            ::grpc::ServerBuilder* builder) {
  return new GrpcWorkerService(env, builder);
}

}  // namespace tensorflow
