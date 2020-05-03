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

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

const int kMaxWorkerRpcRetries = 10;

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            thread::ThreadPool* callback_threadpool,
                            WorkerCacheLogger* logger)
      : channel_(std::move(channel)),
        stub_(channel_),
        cq_(completion_queue),
        callback_threadpool_(callback_threadpool),
        getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
        createworkersession_(Method(GrpcWorkerMethod::kCreateWorkerSession)),
        deleteworkersession_(Method(GrpcWorkerMethod::kDeleteWorkerSession)),
        registergraph_(Method(GrpcWorkerMethod::kRegisterGraph)),
        deregistergraph_(Method(GrpcWorkerMethod::kDeregisterGraph)),
        rungraph_(Method(GrpcWorkerMethod::kRunGraph)),
        cleanupgraph_(Method(GrpcWorkerMethod::kCleanupGraph)),
        cleanupall_(Method(GrpcWorkerMethod::kCleanupAll)),
        recvtensor_(Method(GrpcWorkerMethod::kRecvTensor)),
        recvbuf_(Method(GrpcWorkerMethod::kRecvBuf)),
        logging_(Method(GrpcWorkerMethod::kLogging)),
        tracing_(Method(GrpcWorkerMethod::kTracing)),
        completegroup_(Method(GrpcWorkerMethod::kCompleteGroup)),
        instancesource_(Method(GrpcWorkerMethod::kCompleteInstance)),
        getstepsequence_(Method(GrpcWorkerMethod::kGetStepSequence)),
        markrecvfinished_(Method(GrpcWorkerMethod::kMarkRecvFinished)),
        logger_(logger) {}

  ~GrpcRemoteWorker() override {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
    IssueRequest(request, response, getstatus_, std::move(done), nullptr,
                 fail_fast);
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    IssueRequest(request, response, createworkersession_, std::move(done));
  }

  void DeleteWorkerSessionAsync(CallOptions* call_opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
    IssueRequest(request, response, deleteworkersession_, std::move(done),
                 call_opts);
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

  void RecvBufAsync(CallOptions* call_opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    int64 start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    auto callback = [this, request, response, done, start_usec,
                     logging_active](Status s) {
      if (logging_active) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          RecvBufRespExtra extra;
          response->transport_options().UnpackTo(&extra);
          int64 num_bytes = 0;
          for (const auto& chunk : extra.tensor_content()) {
            num_bytes += chunk.size();
          }
          int64 send_start_usec = start_usec;
          // Prefer start time reported by the sender, if available.
          if (response->send_start_micros()) {
            send_start_usec = std::max(
                start_usec, static_cast<int64>(response->send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->buf_rendezvous_key();
          logger_->RecordDataTransfer(
              step_id, send_start_usec, end_usec, key, request->src_device(),
              request->dst_device(), num_bytes, "", "RecvBuf");
        }
        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->DebugString();
      }

      // Note done() can delete this worker object, so we need to call done()
      // last.
      if (response->require_ack()) {
        IssueMarkRecvFinishedRequest(request->request_id());
      }
      done(s);
    };

    IssueRequest(request, response, recvbuf_, callback, call_opts);
  }

  void CompleteGroupAsync(CallOptions* call_opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    IssueRequest(request, response, completegroup_, std::move(done), call_opts,
                 /*fail_fast=*/false);
  }

  void CompleteInstanceAsync(CallOptions* call_opts,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    IssueRequest(request, response, instancesource_, std::move(done),
                 call_opts);
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
    IssueRequest(request, response, getstepsequence_, std::move(done));
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64 start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    auto callback = [this, request, response, done, start_usec,
                     logging_active](Status s) {
      if (logging_active) {
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
      }

      // Note done() can delete this worker object, so we need to call done()
      // last.
      if (response->metadata().require_ack()) {
        IssueMarkRecvFinishedRequest(request->request_id());
      }
      done(s);
    };

    IssueRequest(request, response, recvtensor_, callback, call_opts);
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
  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response, const ::grpc::string& method,
                    StatusCallback done, CallOptions* call_opts = nullptr,
                    bool fail_fast = true) {
    new RPCState<protobuf::Message>(
        &stub_, cq_, method, *request, response, std::move(done), call_opts,
        callback_threadpool_, /*max_retries=*/0, fail_fast);
  }

  void IssueRequest(const protobuf::Message* request, TensorResponse* response,
                    const ::grpc::string& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    new RPCState<TensorResponse>(&stub_, cq_, method, *request, response,
                                 std::move(done), call_opts,
                                 callback_threadpool_);
  }

  void IssueMarkRecvFinishedRequest(int64 request_id) {
    VLOG(2) << "Send MarkRecvFinishedRequest for request " << request_id;
    MarkRecvFinishedRequest request;
    request.set_request_id(request_id);

    MarkRecvFinishedResponse* response = new MarkRecvFinishedResponse();
    auto done = [response](Status status) { delete response; };
    IssueRequest(&request, response, markrecvfinished_, done);
  }

  // Helper function for initializing the RpcMethod objects below.
  const char* Method(GrpcWorkerMethod id) { return GrpcWorkerMethodName(id); }

  SharedGrpcChannelPtr channel_;
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  thread::ThreadPool* callback_threadpool_;

  const ::grpc::string getstatus_;
  const ::grpc::string createworkersession_;
  const ::grpc::string deleteworkersession_;
  const ::grpc::string registergraph_;
  const ::grpc::string deregistergraph_;
  const ::grpc::string rungraph_;
  const ::grpc::string cleanupgraph_;
  const ::grpc::string cleanupall_;
  const ::grpc::string recvtensor_;
  const ::grpc::string recvbuf_;
  const ::grpc::string logging_;
  const ::grpc::string tracing_;
  const ::grpc::string completegroup_;
  const ::grpc::string instancesource_;
  const ::grpc::string getstepsequence_;
  const ::grpc::string markrecvfinished_;

  // Support for logging.
  WorkerCacheLogger* logger_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
};

WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     thread::ThreadPool* callback_threadpool,
                                     WorkerCacheLogger* logger) {
  return new GrpcRemoteWorker(std::move(channel), completion_queue,
                              callback_threadpool, logger);
}

}  // namespace tensorflow
