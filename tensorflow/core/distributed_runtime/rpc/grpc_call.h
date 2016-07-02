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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_

#include "tensorflow/core/platform/macros.h"

#include "grpc++/grpc++.h"
#include "grpc++/server_builder.h"

namespace tensorflow {

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

  // This method will be called when the response has been sent by
  // `service` and the call is no longer used.
  //
  // `ok` is true if the response sending completed as a "regular
  // event", otherwise it is false.
  void ResponseSent(Service* service, bool ok) {}

  // This method will be called either (i) when the server is notified
  // that the request has been cancelled, or (ii) when the request completes
  // normally. The implementation should distinguish these cases by querying
  // the `grpc::ServerContext` associated with the request.
  virtual void RequestCancelled(Service* service, bool ok) = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback
  // for an incoming RPC.  A Tag owns a reference on the corresponding
  // Call object.
  class Tag {
   public:
    using Callback = void (UntypedCall::*)(Service*, bool);

    // Creates a new `Tag` for the given `UntypedCall`. When the
    // request associated with this tag is complete, `callback` will
    // be called.
    Tag(UntypedCall* call, Callback callback)
        : call_(call), callback_(callback) {
      call_->Ref();
    }

    ~Tag() { call_->Unref(); }

    // Calls the callback associated with this tag.
    //
    // The callback takes ownership of `this->call_`.
    void OnCompleted(Service* service, bool ok) {
      (call_->*callback_)(service, ok);
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
    responder_.Finish(response, status,
                      new typename UntypedCall<Service>::Tag(
                          this, &UntypedCall<Service>::ResponseSent));
    this->Unref();
  }

  void RequestCancelled(Service* service, bool ok) override {
    if (ctx_.IsCancelled()) {
      mutex_lock l(mu_);
      if (cancel_callback_) {
        cancel_callback_();
      }
    }
    // NOTE(mrry): This can be called before or after RequestReceived, so we
    // release `cancel_tag_` (in order to allow the event loop to free it).
    cancel_tag_.release();
  }

  // Registers `callback` as the function that should be called if and when this
  // call is cancelled by the client.
  void SetCancelCallback(std::function<void()> callback) {
    mutex_lock l(mu_);
    cancel_callback_ = callback;
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

    (grpc_service->*enqueue_function)(
        &call->ctx_, &call->request, &call->responder_, cq, cq,
        new typename UntypedCall<Service>::Tag(
            call, &UntypedCall<Service>::RequestReceived));
    call->Unref();
  }

  RequestMessage request;
  ResponseMessage response;

 private:
  // Creates a completion queue tag for handling cancellation by the client.
  // NOTE: This method must be called before this call is enqueued on a
  // completion queue.
  void RegisterCancellationHandler() {
    cancel_tag_.reset(new typename UntypedCall<Service>::Tag(
        this, &UntypedCall<Service>::RequestCancelled));
    ctx_.AsyncNotifyWhenDone(cancel_tag_.get());
  }

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;
  mutex mu_;
  std::function<void()> cancel_callback_ GUARDED_BY(mu_);

  // This tag is initially owned by `*this` and borrowed by
  // `ctx_->AsyncNotifyWhenDone()`. Ownership is transferred to the
  // appropriate service's completion queue after
  // `this->RequestReceived(..., true)` is called.
  std::unique_ptr<typename UntypedCall<Service>::Tag> cancel_tag_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
