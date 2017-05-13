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

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/macros.h"

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/service_type.h"
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

  // This method will be called either (i) when the server is notified
  // that the request has been cancelled, or (ii) when the request completes
  // normally. The implementation should distinguish these cases by querying
  // the `grpc::ServerContext` associated with the request.
  virtual void RequestCancelled(Service* service, bool ok) = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback
  // for an incoming RPC.  An active Tag owns a reference on the corresponding
  // Call object.
  class Tag {
   public:
    // One enum value per supported callback.
    enum Callback { kRequestReceived, kResponseSent, kCancelled };

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

    // Calls the callback associated with this tag.
    //
    // The callback takes ownership of `this->call_`.
    void OnCompleted(Service* service, bool ok) {
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
  // call is cancelled by the client.
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
  std::function<void()> cancel_callback_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
