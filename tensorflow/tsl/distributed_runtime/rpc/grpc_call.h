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

#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_

#include "grpcpp/completion_queue.h"
#include "grpcpp/impl/service_type.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/async_stream.h"
#include "grpcpp/support/async_unary_call.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/refcount.h"

namespace tsl {

// CALL STRUCTURES
// ===============
//
// Each pending (incoming) request corresponds to a call object that
// encapsulates the state of the call. Templates and
// pointers-to-member functions are used to avoid boilerplate and
// redundant closure creation. The class hierarchy is as follows:
//
// * `UntypedCall<Service>`: The base class represents a call that
//   could be associated with any of the methods on a service of type
//   `Service`. Also defines a `Tag` nested class that can be used as
//   the tag in a `grpc::CompletionQueue`.  Each class that
//   instantiates `Service` should have a completion queue polling
//   loop that knows about `UntypedCall<Service>::Tag` objects, and
//   invokes their `OnCompleted()` method to continue processing.
//
// * `Call<Service, GrpcService, Req, Resp>`: This class extends
//   `UntypedCall<Service>` and is additionally parameterized by the
//   gRPC-generated asynchronous service class, and the request and
//   response message types. It defines the state associated with a
//   call (whose type depends on the message types), and stores a
//   pointer to a `Service::HandleFoo()` handler method. Each
//   `Service::HandleFoo()` method knows about the corresponding
//   `Call` type, in order to access its state, and invoke its
//   `SendResponse()` method.
//
// The lifecycle of a call object is as follows.
//
// 1. A `Service` creates a `Call` for a particular method and
//    enqueues it in its completion queue (via an
//    `UntypedCall<Service>::Tag`).
//
// 2. When the tag is returned from `cq_->Next()`, the
//    `UntypedCall::RequestReceived()` method is invoked and takes
//    ownership of the call object. This indirectly invokes the
//    appropriate handler method on `Service`.
//
// 3. After the response has been written (perhaps in another thread),
//    the `Call::SendResponse()` method is invoked. It transfers
//    ownership of the call object back to the completion queue (via
//    an `UntypedCall::Tag`).
//
// 4. When the response has been sent, the tag is returned from
//    `cq_->Next()`, and the call object is deleted.
//

template <class Service>
class GrpcCallTag {
 public:
  virtual ~GrpcCallTag() {}

  // Calls the callback associated with this tag.
  virtual void OnCompleted(Service* service, bool ok) = 0;
};

// Represents a pending request with unknown message types.
template <class Service>
class UntypedCall : public core::RefCounted {
 public:
  virtual ~UntypedCall() {}

  // The implementation of this method should use `service` to handle
  // an incoming request, and (perhaps asynchronously) send the
  // response.
  //
  // One reference on `this` is transferred to the callee, and the
  // callee is responsible for releasing it (typically via
  // `Call::SendResponse()`).
  //
  // `ok` is true if the request was received in a "regular event",
  // otherwise false.
  virtual void RequestReceived(Service* service, bool ok) = 0;

  // This method will be called either (i) when the server is notified
  // that the request has been canceled, or (ii) when the request completes
  // normally. The implementation should distinguish these cases by querying
  // the `grpc::ServerContext` associated with the request.
  virtual void RequestCancelled(Service* service, bool ok) = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback
  // for an incoming RPC.  An active Tag owns a reference on the corresponding
  // Call object.
  class Tag : public GrpcCallTag<Service> {
   public:
    // One enum value per supported callback.
    enum Callback { kRequestReceived, kResponseSent, kCancelled };

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

    // Calls the callback associated with this tag.
    //
    // The callback takes ownership of `this->call_`.
    void OnCompleted(Service* service, bool ok) override {
      switch (callback_) {
        case kRequestReceived:
          call_->RequestReceived(service, ok);
          break;
        case kResponseSent:
          // No special handling needed apart from the Unref below.
          break;
        case kCancelled:
          call_->RequestCancelled(service, ok);
          break;
      }
      call_->Unref();  // Ref acquired when tag handed to grpc.
    }

   private:
    UntypedCall* const call_;  // `this` owns one reference.
    Callback callback_;
  };
};

// Represents a pending call with known request and response message
// types, and a known request-handling method.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class Call : public UntypedCall<Service> {
 public:
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

  Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function), responder_(&ctx_) {}

  virtual ~Call() {}

  void RequestReceived(Service* service, bool ok) override {
    if (ok) {
      this->Ref();
      (service->*handle_request_function_)(this);
    }
  }

  void SendResponse(::grpc::Status status) {
    this->Ref();  // Ref for grpc; released in Tag callback.
    responder_.Finish(response, status, &response_sent_tag_);
    this->Unref();
  }

  void RequestCancelled(Service* service, bool ok) override {
    if (ctx_.IsCancelled()) {
      mutex_lock l(mu_);
      if (cancel_callback_) {
        cancel_callback_();
      }
    }
  }

  // Registers `callback` as the function that should be called if and when this
  // call is canceled by the client.
  void SetCancelCallback(std::function<void()> callback) {
    mutex_lock l(mu_);
    cancel_callback_ = std::move(callback);
  }

  // Clears any cancellation callback that has been registered for this call.
  void ClearCancelCallback() {
    mutex_lock l(mu_);
    cancel_callback_ = nullptr;
  }

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `enqueue_function`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function,
                             bool supports_cancel) {
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
  }

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `method_id`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequestForMethod(
      GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
      int method_id, HandleRequestFunction handle_request_function,
      bool supports_cancel) {
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                    &call->responder_, cq, cq,
                                    &call->request_received_tag_);
  }

  RequestMessage request;
  ResponseMessage response;

  const std::multimap<::grpc::string_ref, ::grpc::string_ref>& client_metadata()
      const {
    return ctx_.client_metadata();
  }

 private:
  // Creates a completion queue tag for handling cancellation by the client.
  // NOTE: This method must be called before this call is enqueued on a
  // completion queue.
  void RegisterCancellationHandler() {
    this->Ref();  // Ref for grpc; released in Tag callback.
    ctx_.AsyncNotifyWhenDone(&cancelled_tag_);
  }

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;

  // Used as void* completion markers from grpc to indicate different
  // events of interest for a Call.
  typedef typename UntypedCall<Service>::Tag Tag;
  Tag request_received_tag_{this, Tag::kRequestReceived};
  Tag response_sent_tag_{this, Tag::kResponseSent};
  Tag cancelled_tag_{this, Tag::kCancelled};

  mutex mu_;
  std::function<void()> cancel_callback_ TF_GUARDED_BY(mu_);
};

// Lifetime of a server-side bidirectional streaming call:
// - The call is created in the static EnqueueRequest method. It transfers
//   ownership to the kCallOpen tag pushed onto the completion queue.
// - If kCallOpen completes successfully, a read is requested and the
//   kRequestReceived tag takes ownership of the call. If kCallOpen fails,
//   e.g. server is shutdown, no further requests are pushed and the call is
//   destroyed (at the end of Tag::OnCompleted).
// - When the first request is received, we Ref() the call and invoke the
//   handler method thereby transferring ownership to the handler method.
//   The handler is responsible for calling SendResponse() or Finish() on this
//   call.
//   - If the handler calls Finish(), e.g. the request was invalid, Finish()
//     transfers ownership from the handler to the kServerFinished tag that
//     it pushes on the completion queue. The ownership is transferred because
//     the ref count is not incremented before putting the tag on the queue.
//   - If the handler calls SendResponse(), SendResponse() transfers ownership
//     to the kResponseSent tag.
// - When kResponseSent completes, we request a new read, which owns the call
//   now.
// - When the next request is received, it is handled the same way as the first
//   request.
//
// Because we request a read only after the write is sent, we can safely reuse
// the same request and response messages for the whole call.
template <class Service>
class ServerUntypedBidirectionalStreamingCall : public core::RefCounted {
 public:
  virtual void RequestReceived(Service* service) = 0;

  // Enqueues a request on the completion queue to read the next request.
  virtual void CallOpen() = 0;

  virtual void RequestRead() = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback.
  // An active Tag owns a reference on the corresponding Call object.
  class Tag : public GrpcCallTag<Service> {
   public:
    // One enum value per supported callback.
    enum class TagType {
      kCallOpen,
      kRequestReceived,
      kResponseSent,
      kServerFinished,
    };

    Tag(ServerUntypedBidirectionalStreamingCall* call, TagType cb)
        : call_(call), callback_(cb) {}

    // Calls the callback associated with this tag and Unrefs this->call_.
    void OnCompleted(Service* service, bool ok) override {
      switch (callback_) {
        case TagType::kCallOpen:
          // Non-ok value indicates that the server has been shutdown before we
          // received a message for this call type. We do nothing to let this
          // call object be destroyed and avoid enqueuing request for another
          // call.
          if (ok) {
            call_->CallOpen();
          }
          break;
        case TagType::kRequestReceived:
          // Non-ok value from completion queue here means that we will not
          // receive any more messages from the client, e.g. the client called
          // WritesDone. There is nothing we need to do in this case. The call
          // will be Unref'ed and deleted. If the client wants to open a new
          // call, we have already enqueued a request for a new call in CallOpen
          // above.
          if (ok) {
            call_->RequestReceived(service);
          }
          break;
        case TagType::kResponseSent:
          if (ok) {
            // The obvious place to request a read would be at the end of
            // RequestReceived(). Unfortunately, this can result in multiple
            // outstanding write requests in the completion queue. This is
            // currently not supported by gRPC, which requires at most one
            // outstanding write request in the completion queue.
            // Requesting a read here, in ResponseSent, works because at
            // this point, the completion queue has no write requests
            // (kResponseSent happens when a write completes).
            // This might be synchronizing the processing more than strictly
            // necessary, but is probably fine because, AFAICT from gRPC docs,
            // the write request completes as soon as it can be written to
            // outgoing buffer.
            call_->RequestRead();
          }
          // ok == false means that the response is not going on the wire
          // because the call is already dead (i.e., canceled, deadline
          // expired, other side dropped the channel, etc). Since the call is
          // dead, there is nothing for us to do, we just let the call be
          // deleted.
          break;
        case TagType::kServerFinished:
          // Whether our finish request is successful or not (whether it went
          // on the wire towards the client), there is nothing for us to do.
          // In the current implementation, there can be no read or write
          // requests in the completion queue (see the comment in kResponseSent)
          // above. Even if there were pending requests, they would complete
          // with a non-ok status, we would not do anything, and let the call be
          // deleted.
          break;
      }
      call_->Unref();  // Ref acquired when tag was handed to grpc.
    }

   private:
    ServerUntypedBidirectionalStreamingCall* const
        call_;  // `this` owns one reference.
    TagType callback_;
  };
};

// Represents a pending call with known request and response message
// types, and a known request-handling method.
// Common usage pattern is to have a single thread waiting on events from
// completion queue and calling Tag::OnCompleted(), which invokes methods
// on this.
// This implementation assumes that the server will generate a single response
// message for each request message. More precisely, this class expects that
// each time it invokes handle_request_function_, the service implementation
// will either call SendResponse or Finish exactly once.
// Not thread-safe.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class ServerBidirectionalStreamingCall
    : public ServerUntypedBidirectionalStreamingCall<Service> {
 public:
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*,
      ::grpc::ServerAsyncReaderWriter<ResponseMessage, RequestMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      ServerBidirectionalStreamingCall<Service, GrpcService, RequestMessage,
                                       ResponseMessage>*);

  ServerBidirectionalStreamingCall(
      HandleRequestFunction handle_request_function, GrpcService* grpc_service,
      ::grpc::ServerCompletionQueue* cq, EnqueueFunction enqueue_function)
      : handle_request_function_(handle_request_function),
        stream_(&ctx_),
        grpc_service_(grpc_service),
        cq_(cq),
        enqueue_function_(enqueue_function) {
    VLOG(3) << "Creating ServerBidirectionalStreamingCall " << this;
  }

  ~ServerBidirectionalStreamingCall() override {
    VLOG(3) << "Destroying ServerBidirectionalStreamingCall " << this;
  }

  void CallOpen() override {
    // Let gRPC know that we can accept another call.
    ServerBidirectionalStreamingCall<
        Service, GrpcService, RequestMessage,
        ResponseMessage>::EnqueueRequest(grpc_service_, cq_, enqueue_function_,
                                         handle_request_function_);
    RequestRead();
  }

  void RequestRead() override {
    this->Ref();
    request_.Clear();
    stream_.Read(&request_, &request_received_tag_);
  }

  void RequestReceived(Service* service) override {
    this->Ref();
    // Request handling should result in a call to SendResponse or Finish.
    (service->*handle_request_function_)(this);
  }

  void SendResponse() {
    // Transferring ownership of this to the response_sent_tag_.
    stream_.Write(response_, &response_sent_tag_);
    // stream_.Write does not save references to response_. We are free to muck
    // around with it as soon as Write returns.
    // We clear the response_ to prepare it for the next response.
    response_.Clear();
  }

  void Finish(::grpc::Status status) {
    // Transferring ownership of this to the server_finished_tag_.
    stream_.Finish(status, &server_finished_tag_);
  }

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `enqueue_function`.
  //
  // The request will be handled by the given `handle_request_function`.
  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function) {
    auto call =
        new ServerBidirectionalStreamingCall<Service, GrpcService,
                                             RequestMessage, ResponseMessage>(
            handle_request_function, grpc_service, cq, enqueue_function);

    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->stream_, cq, cq,
                                      &call->call_open_tag_);
  }

  const RequestMessage& request() const { return request_; }
  ResponseMessage* mutable_response() { return &response_; }

 private:
  // Request and response messages are reused for each request/response exchange
  // between the client and the server.
  RequestMessage request_;
  ResponseMessage response_;
  ::grpc::ServerContext ctx_;

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerAsyncReaderWriter<ResponseMessage, RequestMessage> stream_;

  // Used as void* completion markers from grpc to indicate different
  // events of interest for a ServerBidirectionalStreamingCall.
  typedef typename ServerUntypedBidirectionalStreamingCall<Service>::Tag Tag;
  // At most one tag of each kind may be given to gRPC at any one time.
  // Beyond semantic sanity, this is needed to ensure proper ref counting
  // of this call object.
  Tag call_open_tag_{this, Tag::TagType::kCallOpen};
  Tag request_received_tag_{this, Tag::TagType::kRequestReceived};
  Tag response_sent_tag_{this, Tag::TagType::kResponseSent};
  Tag server_finished_tag_{this, Tag::TagType::kServerFinished};

  // These fields are used only to spawn another instance of this to accept
  // more streaming calls.
  GrpcService* grpc_service_;
  ::grpc::ServerCompletionQueue* cq_;
  EnqueueFunction enqueue_function_;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
