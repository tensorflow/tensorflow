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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"

#include <utility>

#include "grpc++/grpc++.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            WorkerCacheLogger* logger)
      : channel_(std::move(channel)),
        cq_(completion_queue),
        getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
        createworkersession_(Method(GrpcWorkerMethod::kCreateWorkerSession)),
        registergraph_(Method(GrpcWorkerMethod::kRegisterGraph)),
        deregistergraph_(Method(GrpcWorkerMethod::kDeregisterGraph)),
        rungraph_(Method(GrpcWorkerMethod::kRunGraph)),
        cleanupgraph_(Method(GrpcWorkerMethod::kCleanupGraph)),
        cleanupall_(Method(GrpcWorkerMethod::kCleanupAll)),
        recvtensor_(Method(GrpcWorkerMethod::kRecvTensor)),
        logging_(Method(GrpcWorkerMethod::kLogging)),
        tracing_(Method(GrpcWorkerMethod::kTracing)),
        logger_(logger) {}

  ~GrpcRemoteWorker() override {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    IssueRequest(request, response, getstatus_, std::move(done));
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    IssueRequest(request, response, createworkersession_, std::move(done));
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    IssueRequest(request, response, registergraph_, std::move(done));
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    IssueRequest(request, response, deregistergraph_, std::move(done));
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
    IssueRequest(request, response, rungraph_, std::move(done), call_opts);
  }
  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    IssueRequest(&request->ToProto(), get_proto_from_wrapper(response),
                 rungraph_, std::move(done), call_opts);
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    IssueRequest(request, response, cleanupgraph_, std::move(done));
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    IssueRequest(request, response, cleanupall_, std::move(done));
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64 start_usec = Env::Default()->NowMicros();
    // Don't propagate dma_ok over gRPC.
    RecvTensorRequest* req_copy = nullptr;
    if (request->dma_ok()) {
      req_copy = new RecvTensorRequest;
      *req_copy = *request;
      req_copy->set_dma_ok(false);
    }
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active && req_copy == nullptr) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else if (!logging_active) {
      wrapper_done = [req_copy, done](Status s) {
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    } else {
      wrapper_done = [this, request, req_copy, response, done,
                      start_usec](Status s) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          int64 bytes = response->tensor().TotalBytes();
          int64 send_start_usec = start_usec;
          // If a send start time was reported by the other side, use
          // that instead.  Maybe we should mark the display if we're using
          // our local time instead of the remote start time?
          if (response->metadata().send_start_micros()) {
            // send_start_micros is the timestamp taken when the
            // remote machine began to send the RecvTensor response.
            // Due to clock skew between source and dest machines, it
            // is possible that send_start_micros can be larger than
            // end_usec or less than start_usec.
            //
            // To respect causality, we enforce the invariants that
            // the RecvTensor response can not have been sent before
            // the RecvTensor request, and must have been sent before
            // it was received.
            send_start_usec = std::max(
                start_usec,
                static_cast<int64>(response->metadata().send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->rendezvous_key();
          std::vector<string> key_parts = str_util::Split(key, ';');
          if (key_parts.size() != 5) {
            LOG(WARNING) << "Bad key: " << key;
          } else {
            logger_->RecordRecvTensor(step_id, send_start_usec, end_usec,
                                      key_parts[3],  // tensor name
                                      key_parts[0],  // src_device
                                      key_parts[2],  // dst_device
                                      bytes);
          }
        }
        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->metadata().DebugString();
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(req_copy ? req_copy : request, response, recvtensor_,
                 *cb_to_use, call_opts);
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    IssueRequest(request, response, logging_, done);
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    IssueRequest(request, response, tracing_, done);
  }

 private:
  // Object allocated per active RPC.
  template <class RequestMessage, class ResponseMessage>
  class RPCState final : public GrpcClientCQTag {
   public:
    RPCState(::grpc::ChannelInterface* channel, ::grpc::CompletionQueue* cq,
             const ::grpc::RpcMethod& method, const RequestMessage& request,
             StatusCallback done, CallOptions* call_opts)
        : call_opts_(call_opts),
          reader_(channel, cq, method, InitContext(call_opts), request),
          done_(std::move(done)) {}

    ~RPCState() override {}

    void StartRPC(ResponseMessage* response) {
      reader_.Finish(response, &status_, this);
    }

    void OnCompleted(bool ok) override {
      if (!ok) {
        VLOG(2) << "Call returned with non-ok status: "
                << status_.error_message();
      }
      if (call_opts_) {
        call_opts_->ClearCancelCallback();
      }
      done_(FromGrpcStatus(status_));
      delete this;
    }

   private:
    CallOptions* call_opts_;
    ::grpc::ClientContext context_;
    ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
    ::grpc::Status status_;
    StatusCallback done_;

    ::grpc::ClientContext* InitContext(CallOptions* call_opts) {
      // The initialization and recovery protocols rely on blocking
      // until we get a response.
      context_.set_fail_fast(false);
      if (call_opts) {
        call_opts->SetCancelCallback([this]() { context_.TryCancel(); });
      }
      return &context_;
    }
  };

  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  template <class RequestMessage, class ResponseMessage>
  void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                    const ::grpc::RpcMethod& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    auto state = new RPCState<RequestMessage, ResponseMessage>(
        channel_.get(), cq_, method, *request, std::move(done), call_opts);
    state->StartRPC(response);
  }

  // Helper function for initializing the RpcMethod objects below.
  ::grpc::RpcMethod Method(GrpcWorkerMethod id) {
    return ::grpc::RpcMethod(GrpcWorkerMethodName(id),
                             ::grpc::RpcMethod::NORMAL_RPC, channel_);
  }

  SharedGrpcChannelPtr channel_;
  ::grpc::CompletionQueue* cq_;

  const ::grpc::RpcMethod getstatus_;
  const ::grpc::RpcMethod createworkersession_;
  const ::grpc::RpcMethod registergraph_;
  const ::grpc::RpcMethod deregistergraph_;
  const ::grpc::RpcMethod rungraph_;
  const ::grpc::RpcMethod cleanupgraph_;
  const ::grpc::RpcMethod cleanupall_;
  const ::grpc::RpcMethod recvtensor_;
  const ::grpc::RpcMethod logging_;
  const ::grpc::RpcMethod tracing_;

  // Support for logging.
  WorkerCacheLogger* logger_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
};

WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     WorkerCacheLogger* logger) {
  return new GrpcRemoteWorker(std::move(channel), completion_queue, logger);
}

}  // namespace tensorflow
