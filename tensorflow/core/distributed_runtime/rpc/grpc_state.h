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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_

#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {

// Object allocated per active RPC.
template <class Response>
class RPCState : public GrpcClientCQTag {
 public:
  // Default behavior is to set fail_fast = False and handle timeouts manually.
  RPCState(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
           const ::grpc::string& method, const protobuf::Message& request,
           Response* response, StatusCallback done, CallOptions* call_opts,
           thread::ThreadPool* threadpool)
      : RPCState(stub, cq, method, request, response, std::move(done),
                 call_opts, threadpool, /*fail_fast=*/false,
                 /*timeout_in_ms=*/0) {}

  template <typename Request>
  RPCState(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
           const ::grpc::string& method, const Request& request,
           Response* response, StatusCallback done, CallOptions* call_opts,
           thread::ThreadPool* threadpool, bool fail_fast, int64 timeout_in_ms)
      : call_opts_(call_opts), threadpool_(threadpool), done_(std::move(done)) {
    context_.set_fail_fast(fail_fast);
    if (timeout_in_ms > 0) {
      context_.set_deadline(gpr_time_from_millis(timeout_in_ms, GPR_TIMESPAN));
    }

    if (call_opts) {
      call_opts->SetCancelCallback([this]() { context_.TryCancel(); });
    }

    response_ = response;
    ::grpc::Status s = GrpcMaybeUnparseProto(request, &request_buf_);
    if (!s.ok()) {
      LOG(ERROR) << "GrpcMaybeUnparseProto returned with non-ok status: "
                 << s.error_message();
    }
    call_ =
        std::move(stub->PrepareUnaryCall(&context_, method, request_buf_, cq));
    call_->StartCall();
    call_->Finish(&response_buf_, &status_, this);
  }

  void OnCompleted(bool ok) override {
    if (call_opts_) {
      call_opts_->ClearCancelCallback();
    }
    Status s = FromGrpcStatus(status_);
    if (s.ok() && !ok) {
      // Since this function is only being used for processing the response
      // to Finish for client-side unary calls, ok should never be false
      s.Update(errors::Internal("unexpected ok value at rpc completion"));
    }

    if (s.ok()) {
      if (threadpool_) {
        // Run parse and callback in another thread, returning this
        // one to service more RPCs.
        threadpool_->Schedule([this]() { ParseAndCallDone(); });
      } else {
        ParseAndCallDone();
        return;
      }
    } else {
      VLOG(2) << "Call returned with non-ok status: " << s;
      done_(s);
      delete this;
    }
  }

  void ParseAndCallDone() {
    Status s;
    if (!GrpcMaybeParseProto(&response_buf_, response_)) {
      s.Update(errors::Internal("could not parse rpc response"));
    }
    done_(s);
    delete this;
  }

 private:
  CallOptions* call_opts_;
  ::grpc::ClientContext context_;
  thread::ThreadPool* threadpool_;
  std::unique_ptr<::grpc::GenericClientAsyncResponseReader> call_;
  Response* response_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  ::grpc::Status status_;
  StatusCallback done_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
