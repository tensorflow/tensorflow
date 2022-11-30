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

#include <queue>
#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_state.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::RPCState;
// NOLINTEND(misc-unused-using-decls)

// Represents state associated with one streaming RPC call.
// Similarly to above, we extract the methods of StreamingRPCState that don't
// need to be templated into this abstract class.
// Currently, *StreamingRPCState does not support client closing the call as
// there is no use case for it - current clients keep the streaming call open
// as long as possible. If/when the need arises, support can be added
// by calling GenericClientAsyncReaderWriter::WritesDone with a new tag
// TagType::kClientFinished and handling the completion in a new callback.
class UntypedStreamingRPCState : public core::RefCounted {
 public:
  virtual void CallStarted(bool ok) = 0;
  virtual void RequestWriteCompleted(bool ok) = 0;
  virtual void ResponseReadCompleted(bool ok) = 0;
  virtual void CallFinished(bool ok) = 0;

  virtual string DebugString() const = 0;

  class Tag : public GrpcClientCQTag {
   public:
    // One enum value per supported callback.
    enum class TagType {
      kCallStarted,
      kRequestWriteCompleted,
      kResponseReadCompleted,
      kCallFinished,
    };

    Tag(UntypedStreamingRPCState* streaming_state, Tag::TagType type);

    // Calls the callback associated with this tag and Unrefs
    // `this->streaming_state_`.
    void OnCompleted(bool ok) override;

   private:
    // OnCompleted() consumes on reference each time it is called.
    UntypedStreamingRPCState* const streaming_state_;
    const Tag::TagType type_;
  };
};

const char* ToString(UntypedStreamingRPCState::Tag::TagType tag_type);

// Represents a single request/response exchange between client and the server.
// A single streaming call contains a sequence of exchanges. Besides the
// messages, exchange contains:
//  - the user callback to invoke when exchange completes (response is received
//    or an error occurs).
//  - The current state of the exchange.
class Exchange {
 public:
  enum class State {
    kExchangeCreated,
    kRequestWriteIssued,
    kRequestWriteCompleted,
    kResponseReadIssued,
  };

  Exchange(const ::grpc::ByteBuffer& request_buf, protobuf::Message* response,
           StatusCallback cb, string debug_string)
      : state_(State::kExchangeCreated),
        request_buf_(request_buf),
        response_(response),
        cb_(std::move(cb)),
        debug_string_(std::move(debug_string)) {}

  const ::grpc::ByteBuffer& request_buf() { return request_buf_; }
  ::grpc::ByteBuffer* response_buf() { return &response_buf_; }

  void MarkRequestWriteIssued() {
    DCHECK(state_ == State::kExchangeCreated);
    state_ = State::kRequestWriteIssued;
  }
  void MarkRequestWriteCompleted() {
    DCHECK(state_ == State::kRequestWriteIssued);
    state_ = State::kRequestWriteCompleted;
  }
  void MarkResponseReadIssued() {
    DCHECK(state_ == State::kRequestWriteCompleted);
    state_ = State::kResponseReadIssued;
  }

  // If `status` is success, completes this exchange by parsing the
  // response_buf_ and invoking cb_ with OkStatus(). Else, invokes the
  // callback with `status`.
  void Complete(Status status);

  const State& state() const { return state_; }

  string DebugString() const;

 private:
  State state_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  protobuf::Message* response_;
  StatusCallback cb_;
  string debug_string_;
};

const char* ToString(Exchange::State s);

std::ostream& operator<<(std::ostream& os, const Exchange::State& state);

// Represents a queue of exchanges.
// When a client sends a new request a new exchange is created and added to the
// end of the queue. Completed exchanges are popped from the front of the queue.
// An explicit exchange queue is needed to brdige the client, which can send new
// requests at any time, with gRPC infrastructure, which can handle a single
// read and a single write request at a time.
//
// As the exchange progresses (request sending initiated, request sending
// completed, response reading initiated) the queue helps to make sure that the
// right operation is issued on the right exchange at the right time.
//
// To satisfy gRPC constraints, the states of exchanges must be as follows
// starting from the front of the queue:
//  - 0 or 1 exchange in kResponseReadIssued state
//  - 0 or more exchanges in kRequestWriteCompleted state
//  - 0 or 1 exchange in kRequestWriteIssued state
//  - 0 or more exchanges in kExchangeCreated state
//
// Thread-compatible.
class ExchangeQueue {
 public:
  // Creates a new exchange and adds it to the end of the queue.
  void Emplace(const ::grpc::ByteBuffer& request_buf,
               protobuf::Message* response, StatusCallback cb,
               std::string debug_string);

  // Returns an exchange for which we can initiate request writing, if any.
  // Returns nullptr if there is no such exchange.
  Exchange* GetReadyForRequestWriting();

  // Returns an exchange for which we can initiate response reading, if any.
  // Returns nullptr if there is no such exchange.
  Exchange* GetReadyForResponseReading();

  // Changes the state of the exchange that is current in kRequestWriteIssued
  // state to kRequestWriteCompleted state.
  // REQUIRES: There is an exchange in kRequestWriteIssued state.
  void MarkRequestWriteCompleted();

  // Returns the exchange at the front of the queue.
  // REQUIRES: ExchangeQueue is not empty.
  Exchange& GetFront();

  // Removes the exchange at the front of the queue.
  // REQUIRES: ExchangeQueue is not empty.
  void PopFront();

  // Returns a string containing addresses and states of all exchanges in this
  // queue.
  string DebugString() const;

  // Swaps the contents of this and `other`.
  void Swap(ExchangeQueue* other);

  // Completes all exchanges in this with `status`.
  void CompleteAll(Status status);

  void CallStarted() { call_started_ = true; }

 private:
  // Does nothing by default. Turn on VLOG(5) to enable.
  // Checks that this ExchangeQueue is in a valid state.
  // Kills the process if not.
  void CheckInvariants();

  // We can't process any exchanges until the call has started.
  bool call_started_ = false;

  // std::queue is based on std::deque by default. std::deque provides
  // fairly strong iterator stability.
  std::deque<Exchange> exchanges_;
};  // namespace tensorflow

// Represents state associated with one streaming RPC call.
// Thread-safe
template <class Response>
class StreamingRPCState : public UntypedStreamingRPCState {
 public:
  // Default behavior is to set fail_fast = False and handle timeouts
  // manually.
  StreamingRPCState(
      std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call,
      const std::shared_ptr<::grpc::ClientContext>& context)
      : context_(context), call_(std::move(call)), call_state_(State::kActive) {
    Ref();
    VLOG(3) << "Created new StreamingRPCState " << this;
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::StartCall";
    call_->StartCall(&call_started_tag_);
  }

  ~StreamingRPCState() override {
    VLOG(3) << "Destructing StreamingRPCState " << this;
  }

  // Attempts to send the next request. `done` is invoked when
  // `response` has been filled with the data from the server, or if there
  // is an error. `done` can be invoked before SendNextRequest returns.
  // Return `true` if the call is alive and the `done` callback has or
  // will be invoked. If the call is dead, returns `false`. `done` callback
  // will not be invoked in this case.
  // REQUIRES: The call has been started, i.e. WaitForCallStarted() has
  // returned.
  bool SendNextRequest(const protobuf::Message& request, Response* response,
                       const StatusCallback& done) {
    ::grpc::ByteBuffer request_buf;
    ::grpc::Status s = tsl::GrpcMaybeUnparseProto(request, &request_buf);
    if (!s.ok()) {
      Status status = FromGrpcStatus(s);
      LOG(ERROR) << "GrpcMaybeUnparseProto returned with non-ok status: "
                 << status.ToString();
      done(status);
      return true;
    }

    mutex_lock l(mu_);
    if (call_state_ != State::kActive) {
      // `done` is not invoked intentionally.
      return false;
    }
    if (VLOG_IS_ON(3)) {
      // If vlog 3 is enabled, include first 100 chars of request as debug
      // string.
      exchanges_.Emplace(request_buf, response, done,
                         request.ShortDebugString().substr(0, 100));
    } else {
      exchanges_.Emplace(request_buf, response, done, "");
    }
    MaybeIssueRequestWriteLocked();
    return true;
  }

  void CallStarted(bool ok) override {
    VLOG(3) << "StreamingRPCState(" << this << ")::CallStarted(ok=" << ok
            << ")";
    mutex_lock l(mu_);
    if (!ok) {
      call_state_ = State::kDone;
      return;
    }
    exchanges_.CallStarted();
    // Now that the call has started, we can write our first request, if any.
    MaybeIssueRequestWriteLocked();
  }

  void RequestWriteCompleted(bool ok) override {
    VLOG(3) << "StreamingRPCState(" << this
            << ")::RequestWriteCompleted(ok=" << ok << ")";
    mu_.lock();
    if (call_state_ != State::kActive) {
      mu_.unlock();
      return;
    }
    exchanges_.MarkRequestWriteCompleted();
    // Issue ResponseRead regardless of OK status on completing RequestWrite.
    // If the underlying completion queue is in Not-OK status due to previous
    // request failuress (i.e., `ok` from `Next` call on completion queue is
    // False), delay the error in ResponseRead so we can get the remote error
    // message from response buffer.
    MaybeIssueResponseReadLocked();

    if (ok) {
      MaybeIssueRequestWriteLocked();
    }
    mu_.unlock();
  }

  void ResponseReadCompleted(bool ok) override {
    VLOG(3) << "StreamingRPCState(" << this
            << ")::ResponseReadCompleted(ok=" << ok << ")";
    mu_.lock();
    if (call_state_ != State::kActive) {
      mu_.unlock();
      return;
    }
    if (!ok) {
      IssueCallFinishLocked();
      mu_.unlock();
      return;
    }

    // Complete the exchange without holding the lock because user's
    // callback can call back into this RPC code resulting in a deadlock.
    // No other thread can pop this exchange while we release the lock because
    // this is the only method that pops exchanges and it is called from a
    // single thread that waits on completion queue events.
    Exchange* e;
    e = &exchanges_.GetFront();
    mu_.unlock();

    e->Complete(OkStatus());

    {
      mutex_lock l(mu_);
      exchanges_.PopFront();
      MaybeIssueResponseReadLocked();
    }
  }

  void CallFinished(bool ok) override {
    VLOG(3) << "StreamingRPCState(" << this << ")::CallFinished(ok=" << ok
            << ")";
    mu_.lock();
    DCHECK(call_state_ != State::kActive);
    if (call_state_ != State::kFinishing) {
      mu_.unlock();
      return;
    }

    Status s = FromGrpcStatus(call_status_);
    if (s.ok() && !ok) {
      s.Update(
          errors::Internal("GRPC status is okay but CompletionQueueStatus is "
                           "not.  This should never happen.",
                           context_->debug_error_string()));
    }
    // unlocks mu_
    MarkDoneAndCompleteExchanges(s);
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return exchanges_.DebugString();
  }

 private:
  enum class State {
    kActive,
    kFinishing,
    kDone,
  };

  void MarkDoneAndCompleteExchanges(Status status)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_UNLOCK_FUNCTION(mu_) {
    call_state_ = State::kDone;
    VLOG(2) << "Ending gRPC streaming call on the client side due to "
            << status.ToString();
    // Swap the exchanges_ into a temporary ExchangeQueue so that we can
    // complete all exchanges without holding mu_ in case user callback
    // reach back into this. This should be impossible now, but safer for
    // the future.
    ExchangeQueue queue;
    exchanges_.Swap(&queue);
    mu_.unlock();
    queue.CompleteAll(status);
  }

  void MaybeIssueRequestWriteLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    Exchange* exchange = exchanges_.GetReadyForRequestWriting();
    if (exchange == nullptr) {
      // There are no queued exchanges, there is already an outstanding write,
      // or there are no just created exchanges.
      return;
    }
    exchange->MarkRequestWriteIssued();
    Ref();
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::Write";
    call_->Write(exchange->request_buf(), &request_write_completed_tag_);
  }

  void MaybeIssueResponseReadLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    Exchange* exchange = exchanges_.GetReadyForResponseReading();
    if (exchange == nullptr) {
      return;
    }
    exchange->MarkResponseReadIssued();
    Ref();
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::Read";
    call_->Read(exchange->response_buf(), &response_read_completed_tag_);
  }

  void IssueCallFinishLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    call_state_ = State::kFinishing;
    Ref();
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::Finish";
    // We call finish in response to completed (with error) response reading tag
    // on some exchange. We let this exchange hang in ResponseReadIssued state.
    // ExchangeQueue makes sure that there is at most one exchange in this
    // state. So, no new reads will be issued.
    call_->Finish(&call_status_, &finished_tag_);
  }

  // Holds state for a single request/response exchange between the client
  // and the server.
  typedef typename UntypedStreamingRPCState::Tag Tag;

  // Order of context_ and call_ is important because context_ must outlive
  // call_.
  const std::shared_ptr<const ::grpc::ClientContext> context_;
  std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call_;

  mutable mutex mu_;
  ExchangeQueue exchanges_ TF_GUARDED_BY(mu_);
  State call_state_ TF_GUARDED_BY(mu_);
  ::grpc::Status call_status_ TF_GUARDED_BY(mu_);

  // We can get away with having single instances of these tags per
  // StreamingRPCState because we make sure (as gRPC requires) that
  // there is at most one outstanding Read and at most one outstanding Write
  // in the completion queue.
  // Tags are immutable. No need to guard them.
  Tag call_started_tag_{this, Tag::TagType::kCallStarted};
  Tag request_write_completed_tag_{this, Tag::TagType::kRequestWriteCompleted};
  Tag response_read_completed_tag_{this, Tag::TagType::kResponseReadCompleted};
  Tag finished_tag_{this, Tag::TagType::kCallFinished};
};

// Creates streaming calls and dispatches requests to them.
// In the common case, the client would create a StreamingRPCDispatcher for
// each bidirectional streaming RPC it might want to make. The first time, it
// calls SendNextRequest, a streaming call is initiated and the request is
// sent within this call. Initiation of the call blocks the client. If there are
// no errors, subsequent calls to SendNextRequest would use the already active
// call. If there was an error, the call object will be destroyed after all
// the callbacks for outstanding requests have been invoked. The next call to
// SendNextRequest will initiate a new call.
//
// Callbacks that are part of the same call, are invoked in the order they were
// provided, but callbacks across calls (a failed and a new one) can be invoked
// in any order.
//
// Thread-safe.
template <class Response>
class StreamingRPCDispatcher {
 public:
  StreamingRPCDispatcher(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
                         const ::grpc::string& method)
      : stub_(stub), cq_(cq), method_(method) {}

  // Attempts to send the next request. If there is no active streaming call,
  // starts one and sends the request on top of it. `done` is invoked when
  // `response` has been filled with the data from the server, or if there
  // is an error. `done` can be invoked before SendNextRequest returns.
  void SendNextRequest(const protobuf::Message& request, Response* response,
                       StatusCallback done) {
    mutex_lock l(mu_);
    if (state_ == nullptr) {
      CreateStreamingState();
    }

    bool is_call_alive = state_->SendNextRequest(request, response, done);
    if (is_call_alive) {
      return;
    }

    // The attempt to send failed because the call was dead, create a new
    // call and try again. When the call is dead SendNextRequest does not call
    // `done`.
    CreateStreamingState();

    is_call_alive = state_->SendNextRequest(request, response, done);
    if (!is_call_alive) {
      // Consider retrying to create and start a call few more times.
      done(errors::Unknown("gRPC call failed right after it was created"));
    }
  }

  // Request to cancel the current streaming call. Non-blocking.
  void CancelCall() {
    mutex_lock l(mu_);
    if (state_ == nullptr) {
      return;
    }
    context_->TryCancel();
    state_ = nullptr;
  }

 private:
  void CreateStreamingState() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // ClientContext cannot be reused across calls.
    context_ = std::make_shared<::grpc::ClientContext>();
    // Don't immediately fail StartCall if the channel is not ready. Wait for
    // the channel to become ready.
    context_->set_wait_for_ready(true);

    std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call =
        stub_->PrepareCall(context_.get(), method_, cq_);

    state_.reset(new StreamingRPCState<Response>(std::move(call), context_));
  }

  mutable mutex mu_;

  // Both are thread-safe
  ::grpc::GenericStub* const stub_;
  ::grpc::CompletionQueue* const cq_;

  // Does not need synchronization since it is constant.
  const ::grpc::string method_;

  std::shared_ptr<::grpc::ClientContext> context_ TF_GUARDED_BY(mu_);
  core::RefCountPtr<StreamingRPCState<Response>> state_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
