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

#include "grpc++/generic/generic_stub.h"
#include "grpc++/grpc++.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
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

// Overload of GrpcParseProto so we can decode a TensorResponse without
// extra copying.
bool GrpcParseProto(const ::grpc::ByteBuffer& src, TensorResponse* dst) {
  struct ByteSource : public TensorResponse::Source {
    const ::grpc::ByteBuffer* buffer;
    GrpcByteBufferSource src;
    bool ok;

    ::tensorflow::protobuf::io::ZeroCopyInputStream* contents() override {
      ok = src.Init(*buffer);
      return &src;
    }
  };
  ByteSource bs;
  bs.buffer = &src;
  return dst->ParseFrom(&bs).ok() && bs.ok;
}

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(GrpcCounter* live_rpc_counter,
                            SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            WorkerCacheLogger* logger)
      : counter_(live_rpc_counter),
        channel_(std::move(channel)),
        stub_(channel_),
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
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [this, request, response, done, start_usec](Status s) {
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
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(request, response, recvtensor_, *cb_to_use, call_opts);
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
  template <class ResponseMessage>
  class RPCState : public GrpcClientCQTag {
   public:
    RPCState(GrpcCounter* counter, ::grpc::GenericStub* stub,
             ::grpc::CompletionQueue* cq, const ::grpc::string& method,
             const protobuf::Message& request, ResponseMessage* response,
             StatusCallback done, CallOptions* call_opts)
        : counter_(counter), call_opts_(call_opts), done_(std::move(done)) {
      // TODO(sanjay): The counter will no longer be needed once we
      // get a GenericStub API which allows us to manage an entire
      // RPC with a single completion event instead of four events.
      counter_->Increment();
      // The initialization and recovery protocols rely on blocking
      // until we get a response.
      context_.set_fail_fast(false);
      if (call_opts) {
        call_opts->SetCancelCallback([this]() { context_.TryCancel(); });
      }

      failure_.store(false);
      remaining_callbacks_.store(4);  // Init/Read/Write/Finish callbacks
      response_ = response;
      GrpcUnparseProto(request, &request_buf_);
      // TODO(sanjay): When new enough grpc is available, enable the following:
      //   context_.set_initial_metadata_corked(true);
      // We can then skip the extra state transition for init callback.
      call_ = std::move(stub->Call(&context_, method, cq, this));
      call_initialized_.Notify();
    }

    // Called multiple times: when init done, read done, write done, call done.
    void OnCompleted(bool ok) override {
      if (!ok) failure_.store(true);
      const int old_count = remaining_callbacks_.fetch_sub(1);
      if (old_count > 1) {
        if (old_count == 4) {
          // Init callback finished.  Issue remaining ops.

          // Annoyingly enough, the way the generic call API works is
          // inherently racy.  We can get the following sequence of events:
          //  1. stub->Call() starts.
          //  2. some stuff happens inside grpc
          //  3. grpc delivers the completion event
          //  4. tensorflow event handling thread calls init metadata callback
          //  5. stub->Call() finishes
          //  6. the result of stub->Call() is stored in call_
          // We are currently inside the callback and therefore need to
          // wait for step 6 to finish before attempting to touch call_.
          call_initialized_.WaitForNotification();

          if (ok) {
            // TODO(sanjay): Use WriteLast() when grpc version we are using
            // is new enough.
            call_->Write(request_buf_, this);
            call_->Read(&response_buf_, this);
          } else {
            // Skip Write and Read.
            remaining_callbacks_.fetch_sub(2);
          }
          call_->Finish(&status_, this);
        }
        // Still waiting for some more callbacks to finish.
        return;
      } else {  // old_count == 1, i.e., all callbacks have finished
        // Last callback finished; clean up.
        if (call_opts_) {
          call_opts_->ClearCancelCallback();
        }
        Status s = FromGrpcStatus(status_);
        if (s.ok() && failure_.load()) {
          s.Update(errors::Internal("callback error"));
        }
        if (s.ok() && !GrpcParseProto(response_buf_, response_)) {
          s.Update(errors::Internal("could not parse rpc response"));
        }
        if (!s.ok()) {
          VLOG(2) << "Call returned with non-ok status: " << s;
        }
        done_(s);
        counter_->Decrement();
        delete this;
      }
    }

   private:
    GrpcCounter* const counter_;
    CallOptions* call_opts_;
    ::grpc::ClientContext context_;
    std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call_;
    ResponseMessage* response_;
    ::grpc::ByteBuffer request_buf_;
    ::grpc::ByteBuffer response_buf_;
    ::grpc::Status status_;
    StatusCallback done_;
    std::atomic<bool> failure_;
    std::atomic<int> remaining_callbacks_;
    Notification call_initialized_;
  };

  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response, const ::grpc::string& method,
                    StatusCallback done, CallOptions* call_opts = nullptr) {
    new RPCState<protobuf::Message>(counter_, &stub_, cq_, method, *request,
                                    response, std::move(done), call_opts);
  }
  void IssueRequest(const protobuf::Message* request, TensorResponse* response,
                    const ::grpc::string& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    new RPCState<TensorResponse>(counter_, &stub_, cq_, method, *request,
                                 response, std::move(done), call_opts);
  }

  // Helper function for initializing the RpcMethod objects below.
  const char* Method(GrpcWorkerMethod id) { return GrpcWorkerMethodName(id); }

  GrpcCounter* const counter_;
  SharedGrpcChannelPtr channel_;
  ::grpc::GenericStub stub_;

  ::grpc::CompletionQueue* cq_;

  const ::grpc::string getstatus_;
  const ::grpc::string createworkersession_;
  const ::grpc::string registergraph_;
  const ::grpc::string deregistergraph_;
  const ::grpc::string rungraph_;
  const ::grpc::string cleanupgraph_;
  const ::grpc::string cleanupall_;
  const ::grpc::string recvtensor_;
  const ::grpc::string logging_;
  const ::grpc::string tracing_;

  // Support for logging.
  WorkerCacheLogger* logger_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
};

WorkerInterface* NewGrpcRemoteWorker(GrpcCounter* live_rpc_counter,
                                     SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     WorkerCacheLogger* logger) {
  return new GrpcRemoteWorker(live_rpc_counter, std::move(channel),
                              completion_queue, logger);
}

}  // namespace tensorflow
