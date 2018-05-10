/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_RPC_CALL_CONTAINER_H_
#define TENSORFLOW_CORE_UTIL_RPC_CALL_CONTAINER_H_

#include <list>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {

namespace internal {
// The following class is used for coordination between a `CallContainer`
// instance and a cancellation callback to make sure that the `CallContainer`
// instance waits for the cancellation callback to be destroyed (either because
// a cancellation occurred or because the callback was deregistered) before
// deleting itself. Without this coordination the cancellation callback could
// attempt to access a `CallContainer` instance that is no longer valid.
class NotifyWhenDestroyed {
 public:
  explicit NotifyWhenDestroyed(std::shared_ptr<Notification> notification)
      : notification_(std::move(notification)) {}

  ~NotifyWhenDestroyed() { notification_->Notify(); }

 private:
  std::shared_ptr<Notification> notification_;
};
}  // namespace internal

// The following class is responsible for the life cycle management of a set of
// RPC calls. The calls are started when an instance of the class is created and
// the class contract guarantees to invoke a "done" callback provided by the
// caller when all RPC calls have either completed or been cancelled.
//
// The caller should not make any assumptions about the validity of an instance
// of this class after the provided callback has been invoked, which may be
// immediately after the instance was created.
template <class Call>
class CallContainer {
 public:
  typedef std::function<void(CallContainer<Call>*, int)> CreateCallFn;
  typedef std::function<void(Call*)> StartCallFn;

  // Uses the provided `create_call_fn` and `start_call_fn` functions to create
  // and start a set of RPC calls. When all RPC calls have either completed or
  // been cancelled, the `done` callback is invoked. The caller should not make
  // any assumptions about the validity of the created instance as the instance
  // will delete itself after invoking the `done` callback.
  explicit CallContainer(OpKernelContext* ctx, int num_calls, bool fail_fast,
                         bool try_rpc, AsyncOpKernel::DoneCallback done,
                         CreateCallFn create_call_fn,
                         StartCallFn start_call_fn);

  // Registers a call with this container. This method expects its arguments to
  // match those of a `Call` constructor as it forwards them to an underlying
  // collection, which creates a `Call` instance in place.
  template <class... Args>
  void RegisterCall(Args&&... args);

  // Starts the cancellation of all RPC calls managed by this container.
  void StartCancel();

  // Indicates that the `index`-th RPC call has finished.
  void Done(const Status& s, int index);

 private:
  OpKernelContext* ctx_;
  std::list<Call> calls_;
  const AsyncOpKernel::DoneCallback done_;
  const CancellationToken token_;
  const bool fail_fast_;
  const bool try_rpc_;
  std::shared_ptr<Notification> callback_destroyed_;

  // Performs its own reference counting.
  ReffedStatusCallback* reffed_status_callback_;
};

template <class Call>
CallContainer<Call>::CallContainer(
    OpKernelContext* ctx, int num_calls, bool fail_fast, bool try_rpc,
    AsyncOpKernel::DoneCallback done,
    typename CallContainer<Call>::CreateCallFn create_call_fn,
    typename CallContainer<Call>::StartCallFn start_call_fn)
    : ctx_(ctx),
      done_(std::move(done)),
      token_(ctx->cancellation_manager()->get_cancellation_token()),
      fail_fast_(fail_fast),
      try_rpc_(try_rpc),
      callback_destroyed_(new Notification) {
  CHECK_GT(num_calls, 0);

  // This will run when all RPCs are finished.
  reffed_status_callback_ = new ReffedStatusCallback([this](const Status& s) {
    ctx_->cancellation_manager()->DeregisterCallback(token_);
    ctx_->SetStatus(s);
    done_();
    callback_destroyed_->WaitForNotification();
    delete this;
  });

  // The cancellation callback needs to be registered before the RPC calls are
  // started to make sure that the callback is properly cleaned up by the
  // `reffed_status_callback` when all calls complete. At the same time, the
  // cancellation callback should wait for the RPC calls to be started for the
  // cancellation to take effect.
  std::shared_ptr<internal::NotifyWhenDestroyed> notify_when_destroyed(
      new internal::NotifyWhenDestroyed(callback_destroyed_));
  std::shared_ptr<Notification> calls_started(new Notification);
  bool is_cancelled = !ctx_->cancellation_manager()->RegisterCallback(
      token_, [this, calls_started, notify_when_destroyed]() {
        calls_started->WaitForNotification();
        StartCancel();
      });

  for (int i = 0; i < num_calls; ++i) {
    create_call_fn(this, i);
    // Increase the reference on the callback for each new RPC.
    reffed_status_callback_->Ref();
  }
  for (Call& call : calls_) {
    start_call_fn(&call);
  }
  calls_started->Notify();

  if (is_cancelled) {
    ctx_->SetStatus(errors::Cancelled("Operation has been cancelled."));
    StartCancel();
  }

  // Subtract reference count from the initial creation.
  reffed_status_callback_->Unref();
}

template <class Call>
template <class... Args>
void CallContainer<Call>::RegisterCall(Args&&... args) {
  calls_.emplace_back(std::forward<Args>(args)...);
}

template <class Call>
void CallContainer<Call>::StartCancel() {
  for (auto& call : calls_) {
    call.StartCancel();
  }
}

template <class Call>
void CallContainer<Call>::Done(const Status& s, int index) {
  if (!try_rpc_) {
    reffed_status_callback_->UpdateStatus(s);
  }
  reffed_status_callback_->Unref();
}

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_UTIL_RPC_CALL_CONTAINER_H_
