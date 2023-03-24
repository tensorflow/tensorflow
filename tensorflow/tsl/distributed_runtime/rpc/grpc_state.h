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

#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_

#include <queue>
#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/tsl/distributed_runtime/call_options.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/util/env_var.h"

namespace tsl {

// Object allocated per active RPC.
// Manage the state of a single asynchronous RPC request.  If `max_retries`
// is greater than 0, the request will be retried for any transient failures.
// Note: `parse_proto_fn` is used solely to allow TensorFlow's worker service
// to pass in an optimized function that avoids an unnecessary copy of tensors.
// That is not implemented as an overload of tsl::GrpcMaybeParseProto because it
// has dependencies on many TensorFlow-specific absractions.
template <class Response>
class RPCState : public GrpcClientCQTag {
 public:
  RPCState(
      ::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
      const ::grpc::string& method, const protobuf::Message& request,
      Response* response, StatusCallback done, CallOptions* call_opts,
      thread::ThreadPool* threadpool, int32_t max_retries = 0,
      bool fail_fast = true, const string* target = nullptr,
      std::function<bool(::grpc::ByteBuffer*, Response*)> parse_proto_fn =
          [](::grpc::ByteBuffer* src, Response* dst) {
            return tsl::GrpcMaybeParseProto(src, dst);
          })
      : RPCState(
            stub, cq, method, request, response, std::move(done), call_opts,
            threadpool,
            // 1) If GRPC_FAIL_FAST is set to 'true' or 'false',
            // fail_fast=$GRPC_FAIL_FAST. See b/141948186.
            // 2) Otherwise if GRPC_FAIL_FAST is set to 'use_caller', use the
            // fail_fast from the caller. See b/140260119.
            //
            // Current default: use caller's fail_fast argument.
            //
            // NOTE: Callers mostly set fail_fast=true to prevent job hanging
            // on worker task failures, except a few cases such as GetStatus
            // in cluster initialization and collective param resolution.
            [fail_fast, &done]() -> bool {
              string fail_fast_env;
              TF_CHECK_OK(ReadStringFromEnvVar("GRPC_FAIL_FAST", "use_caller",
                                               &fail_fast_env));
              string fail_fast_env_lower = absl::AsciiStrToLower(fail_fast_env);
              if (fail_fast_env_lower == "true") {
                return true;
              } else if (fail_fast_env_lower == "use_caller") {
                return fail_fast;
              } else if (fail_fast_env_lower == "false") {
                return false;
              } else {
                string error_message = strings::StrCat(
                    "Invalid GRPC_FAIL_FAST config: ", fail_fast_env);
                LOG(WARNING) << error_message;
                done(errors::InvalidArgument(error_message));
                return false;
              }
            }(),
            (call_opts != nullptr ? call_opts->GetTimeout() : 0), max_retries,
            target, parse_proto_fn) {}

  template <typename Request>
  RPCState(
      ::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
      const ::grpc::string& method, const Request& request, Response* response,
      StatusCallback done, CallOptions* call_opts,
      thread::ThreadPool* threadpool, bool fail_fast, int64_t timeout_in_ms,
      int32_t max_retries, const string* target,
      std::function<bool(::grpc::ByteBuffer*, Response*)> parse_proto_fn =
          [](::grpc::ByteBuffer* src, Response* dst) {
            return tsl::GrpcMaybeParseProto(src, dst);
          })
      : call_opts_(call_opts),
        threadpool_(threadpool),
        done_(std::move(done)),
        timeout_in_ms_(timeout_in_ms),
        max_retries_(max_retries),
        cq_(cq),
        stub_(stub),
        method_(method),
        fail_fast_(fail_fast),
        target_(target),
        parse_proto_fn_(std::move(parse_proto_fn)) {
    response_ = response;
    ::grpc::Status s = GrpcMaybeUnparseProto(request, &request_buf_);
    if (!s.ok()) {
      LOG(ERROR) << "GrpcMaybeUnparseProto returned with non-ok status: "
                 << s.error_message();
      // Skip retry logic if we fail to parse our request.
      done_(FromGrpcStatus(s));
      delete this;
      return;
    }
    StartCall();
  }

  void StartCall() {
    context_.reset(new ::grpc::ClientContext());
    context_->set_wait_for_ready(!fail_fast_);
    if (timeout_in_ms_ > 0) {
      context_->set_deadline(
          gpr_time_from_millis(timeout_in_ms_, GPR_TIMESPAN));
    }
    if (call_opts_) {
      call_opts_->SetCancelCallback([this]() { context_->TryCancel(); });
    }

    VLOG(2) << "Starting call: " << method_;

    call_ = stub_->PrepareUnaryCall(context_.get(), method_, request_buf_, cq_);
    call_->StartCall();
    call_->Finish(&response_buf_, &status_, this);
  }

  void OnCompleted(bool ok) override {
    if (call_opts_) {
      call_opts_->ClearCancelCallback();
    }

    VLOG(2) << "Completed call: " << method_;

    Status s = FromGrpcStatus(status_);
    if (s.ok() && !ok) {
      // Since this function is only being used for processing the response
      // to Finish for client-side unary calls, ok should never be false
      s.Update(
          errors::Internal("GRPC status is okay but CompletionQueueStatus is "
                           "not.  This should never happen."));
    }

    if (s.ok()) {
      if (threadpool_) {
        // Run parse and callback in another thread, returning this
        // one to service more RPCs.
        threadpool_->Schedule([this]() { ParseAndCallDone(); });
      } else {
        ParseAndCallDone();
      }
      return;
    }

    VLOG(1) << method_ << " returned with non-ok status: " << s
            << " Retries: " << num_retries_ << " Max: " << max_retries_ << "\n"
            << context_->debug_error_string();
    // Retry if we have any attempts left
    if (++num_retries_ <= max_retries_ &&
        (errors::IsUnavailable(s) || errors::IsUnknown(s))) {
      response_buf_.Clear();
      VLOG(1) << "Retrying call for " << method_ << "Retry: " << num_retries_
              << " of " << max_retries_;

      ComputeRetryBackoffMs(/*min_backoff_ms=*/1, /*max_backoff_ms=*/10000);
      int64_t backoff_us = retry_backoff_ms_ * 1000;
      Env::Default()->SchedClosureAfter(/*micros=*/backoff_us,
                                        [this]() { StartCall(); });
    } else {
      // Attach additional GRPC error information if any to the final status
      string error_msg = s.error_message();
      strings::StrAppend(&error_msg, "\nAdditional GRPC error information");
      if (target_) {
        strings::StrAppend(&error_msg, " from remote target ", *target_);
      }
      strings::StrAppend(&error_msg, ":\n:", context_->debug_error_string());
      s = errors::CreateWithUpdatedMessage(s, error_msg);
      // Always treat gRPC cancellation as a derived error. This ensures that
      // other error types are preferred during status aggregation. (gRPC
      // cancellation messages do not contain the original status message).
      if (s.code() == absl::StatusCode::kCancelled) {
        s = StatusGroup::MakeDerived(s);
      }

      done_(s);
      delete this;
    }
  }

  void ParseAndCallDone() {
    Status s;
    if (!parse_proto_fn_(&response_buf_, response_)) {
      s.Update(errors::Internal("could not parse rpc response"));
    }
    done_(s);
    delete this;
  }

 private:
  void ComputeRetryBackoffMs(int min_backoff_ms, int max_backoff_ms) {
    constexpr float kBackoffBase = 1.3;
    if (retry_backoff_ms_ < 0) {
      retry_backoff_ms_ = min_backoff_ms;
    } else {
      retry_backoff_ms_ *= kBackoffBase;
      if (retry_backoff_ms_ > max_backoff_ms) {
        retry_backoff_ms_ = max_backoff_ms;
      }
    }
  }

  CallOptions* call_opts_;
  std::unique_ptr<::grpc::ClientContext> context_;
  thread::ThreadPool* threadpool_;
  std::unique_ptr<::grpc::GenericClientAsyncResponseReader> call_;
  Response* response_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  ::grpc::Status status_;
  StatusCallback done_;
  int64_t timeout_in_ms_;

  size_t num_retries_ = 0;
  size_t max_retries_;
  double retry_backoff_ms_ = -1;

  ::grpc::CompletionQueue* cq_;
  ::grpc::GenericStub* stub_;
  ::grpc::string method_;
  bool fail_fast_;
  const string* target_;
  std::function<bool(::grpc::ByteBuffer*, Response*)> parse_proto_fn_ = nullptr;
};
}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
