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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_

#include <utility>

#include "grpc++/generic/generic_stub.h"
#include "grpc++/grpc++.h"

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {

// Object allocated per active RPC.
template <class Response>
class RPCState : public GrpcClientCQTag {
 public:
  // Default behavior is to set fail_fast = False and handle timeouts manually.
  RPCState(GrpcCounter* counter, ::grpc::GenericStub* stub,
           ::grpc::CompletionQueue* cq, const ::grpc::string& method,
           const protobuf::Message& request, Response* response,
           StatusCallback done, CallOptions* call_opts)
      : RPCState(counter, stub, cq, method, request, response, std::move(done),
                 call_opts, /*fail_fast=*/false, /*timeout_in_ms=*/0) {}

  template <typename Request>
  RPCState(GrpcCounter* counter, ::grpc::GenericStub* stub,
           ::grpc::CompletionQueue* cq, const ::grpc::string& method,
           const Request& request, Response* response, StatusCallback done,
           CallOptions* call_opts, bool fail_fast, int64 timeout_in_ms)
      : counter_(counter), call_opts_(call_opts), done_(std::move(done)) {
    // TODO(sanjay): The counter will no longer be needed once we
    // get a GenericStub API which allows us to manage an entire
    // RPC with a single completion event instead of four events.
    counter_->Increment();

    context_.set_fail_fast(fail_fast);
    if (timeout_in_ms > 0) {
      context_.set_deadline(gpr_time_from_millis(timeout_in_ms, GPR_TIMESPAN));
    }

    if (call_opts) {
      call_opts->SetCancelCallback([this]() { context_.TryCancel(); });
    }

    failure_.store(false);
    remaining_callbacks_.store(4);  // Init/Read/Write/Finish callbacks
    response_ = response;
    GrpcMaybeUnparseProto(request, &request_buf_);
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
      if (s.ok() && !GrpcMaybeParseProto(response_buf_, response_)) {
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
  Response* response_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  ::grpc::Status status_;
  StatusCallback done_;
  std::atomic<bool> failure_;
  std::atomic<int> remaining_callbacks_;
  Notification call_initialized_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
