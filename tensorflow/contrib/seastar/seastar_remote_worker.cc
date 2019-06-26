#include <utility>

#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_remote_worker.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_interface.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class SeastarRemoteWorker : public WorkerInterface,
                            public SeastarWorkerInterface {
 public:
  explicit SeastarRemoteWorker(seastar::channel* chan,
                               WorkerCacheLogger* logger,
                               WorkerEnv* env)
      : seastar_channel_(chan),
        logger_(logger),
        env_(env) {
  }

  ~SeastarRemoteWorker() override {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    GetStatusAsyncWithOptions(request, response, done, nullptr);
  }

  void GetStatusAsyncWithOptions(const GetStatusRequest* request,
                                 GetStatusResponse* response,
                                 StatusCallback done,
                                 CallOptions* call_opts) {
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kGetStatus,
                   std::move(done),
                   call_opts);
    });
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kCreateWorkerSession,
                   std::move(done));
    });
  }

  void DeleteWorkerSessionAsync(CallOptions* call_opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done, call_opts] {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kDeleteWorkerSession,
                   std::move(done),
                   call_opts);
    });
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kRegisterGraph,
                   std::move(done));
    });
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kDeregisterGraph,
                   std::move(done));
    });
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
    TRACEPRINTF("Seastar RunGraph: %lld", request->step_id());
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kRunGraph,
                   std::move(done),
                   call_opts);
    });
  }

  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    TRACEPRINTF("wrapped Seastar RunGraph: %lld", request->step_id());
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(&request->ToProto(),
                   get_proto_from_wrapper(response),
                   SeastarWorkerServiceMethod::kRunGraph,
                   std::move(done),
                   call_opts);
    });
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kCleanupGraph,
                   std::move(done));
    });
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kCleanupAll,
                   std::move(done));
    });
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("SeastarWorker::RecvTensorAsync()")); 
  }

  void RecvTensorAsync(CallOptions* call_opts,
                       const RecvTensorRequest* request,
                       SeastarTensorResponse* response,
                       StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    // Don't propagate dma_ok over Seastar.
    RecvTensorRequest* req_copy = nullptr;
    if (request->dma_ok()) {
      req_copy = new RecvTensorRequest;
      *req_copy = *request;
      req_copy->set_dma_ok(false);
    }
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (req_copy == nullptr) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [req_copy, done](Status s) {
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(req_copy ? req_copy : request, response,
                 SeastarWorkerServiceMethod::kRecvTensor, 
                 std::move(*cb_to_use), call_opts);
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kLogging,
                   done);
    });
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request,
                   response,
                   SeastarWorkerServiceMethod::kTracing,
                   done);
    });
  }

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::RecvBufAsync()"));
  }

  void CompleteGroupAsync(CallOptions* opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::CompleteGroupAsync()"));
  }

  void CompleteInstanceAsync(CallOptions* ops,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::CompleteInstanceAsync()"));
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::GetStepSequenceAsync()"));
  }

 private:
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response,
                    const SeastarWorkerServiceMethod method,
                    StatusCallback done,
                    CallOptions* call_opts = nullptr) {
      auto tag = new SeastarClientTag(method, env_);
      InitSeastarClientTag(const_cast<protobuf::Message*>(request),
                           response, std::move(done), tag, call_opts);
      tag->StartReq(seastar_channel_);
    }

    void IssueRequest(const protobuf::Message* request,
                      SeastarTensorResponse* response,
                      const SeastarWorkerServiceMethod method,
                      StatusCallback done,
                      CallOptions* call_opts = nullptr) {
      auto tag = new SeastarClientTag(method, env_);
      InitSeastarClientTag(const_cast<protobuf::Message*>(request),
                           response, std::move(done), tag, call_opts);
      tag->StartReq(seastar_channel_);
    }

private:
    seastar::channel* seastar_channel_;
    WorkerCacheLogger* logger_;
    WorkerEnv* env_;
    
    TF_DISALLOW_COPY_AND_ASSIGN(SeastarRemoteWorker);
};

WorkerInterface* NewSeastarRemoteWorker(seastar::channel* seastar_channel,
                                        WorkerCacheLogger* logger,
                                        WorkerEnv* env) {
    return new SeastarRemoteWorker(seastar_channel, logger, env);
}

}  // namespace tensorflow
