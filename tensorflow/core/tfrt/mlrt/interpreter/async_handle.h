/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ASYNC_HANDLE_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ASYNC_HANDLE_H_

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime
#include "tfrt/concurrency/chain.h"  // from @tf_runtime

namespace mlrt {

// mlrt::AsyncHandle is a specialized future for mananging context of an async
// execution.
//
// Example usage:
//
//  // Create the context the async execution by copying the current context.
//  auto [promise, handle] = AsyncHandle::Allocate(current_context);
//
//  // Set up completion signal through the `promise` created.
//  handle.execution_context().set_exit_handler(
//      [promise = std::move(promise)]() { promise.Finish(); });
//
//  // Launch execution.
//  thread_pool.Schedule([&execution_context = handle.execution_context()](){
//    execution_context.Call(...);
//    Execute(execution_context);
//  });
//
//  // Pass `handle` to places that need to wait for the execution.
//  other_execution_context.Await(std::move(handle));
//
class AsyncHandle {
 public:
  class Promise {
   public:
    Promise(const Promise&) = delete;
    Promise& operator=(const Promise&) = delete;
    Promise(Promise&&) = default;
    Promise& operator=(Promise&&) = default;

    ~Promise() {
      DCHECK(!shared_state_ || shared_state_.IsAvailable())
          << "A non-empty promise must be fulfilled.";
    }

    void Finish(absl::Status status) && {
      if (status.ok()) {
        shared_state_.SetStateConcrete();
      } else {
        shared_state_.SetError(std::move(status));
      }
    }

    // We don't need HandleError() method for AsyncHandle::Promise because it is
    // managed by the framework internally and should never be placed in the
    // register.

   private:
    explicit Promise(tsl::AsyncValueRef<tsl::Chain> shared_state)
        : shared_state_(std::move(shared_state)) {}
    tsl::AsyncValueRef<tsl::Chain> shared_state_;

    friend class AsyncHandle;
  };

  // Allocate an AsyncHandle and the corresponding promise.
  static std::pair<Promise, AsyncHandle> Allocate(
      const ExecutionContext& current);

  AsyncHandle(const AsyncHandle&) = delete;
  AsyncHandle& operator=(const AsyncHandle&) = delete;
  AsyncHandle(AsyncHandle&&) = default;
  AsyncHandle& operator=(AsyncHandle&&) = default;

  ~AsyncHandle() {
    CHECK(!shared_state_ || shared_state_.IsAvailable())  // Crash OK
        << "A non-empty AsyncHandle must be awaited.";
  }

  // Then() enqueues a callback which will be called when the future is
  // fulfilled with either an error or a value.
  //
  // The following Then() overloads accept a callback with the following
  // signatures:
  //
  // 1) void(absl::Status)
  //    The argument is the status of this future in ready state.
  //
  // 2) void()
  //    There is no argument. The callback will be called whenever it is ready.

  template <typename F,
            typename Arg = std::decay_t<future_internal::ArgumentType<F>>>
  typename std::enable_if<std::is_same_v<Arg, absl::Status>, void>::type Then(
      F then) && {
    CHECK(shared_state_);  // Crash OK
    auto* shared_state_ptr = shared_state_.GetAsyncValue();
    shared_state_ptr->AndThen([shared_state = std::move(shared_state_),
                               execution_context =
                                   std::move(execution_context_),
                               then = std::move(then)]() mutable {
      future_internal::InvokeThen(std::move(then), shared_state.GetAsyncValue(),
                                  future_internal::ArgTag<Arg>());
    });
  }

  template <typename F,
            typename Arg = std::decay_t<future_internal::ArgumentType<F>>>
  typename std::enable_if<std::is_void_v<Arg>, void>::type Then(F then) && {
    CHECK(shared_state_);  // Crash OK
    auto* shared_state_ptr = shared_state_.GetAsyncValue();
    shared_state_ptr->AndThen(
        [shared_state = std::move(shared_state_),
         execution_context = std::move(execution_context_),
         then = std::move(then)]() mutable { std::move(then)(); });
  }

  void HandleError(Value* arg) {
    if (!shared_state_ || shared_state_.IsAvailable()) {
      // This is an empty handle or it is already finished.
      return;
    }

    auto& execution_context = *arg->Get<ExecutionContext*>();
    execution_context.LogError(absl::InternalError(absl::StrCat(
        "UnwindOnError: unwind AsyncHandle of context ",
        absl::Hex(reinterpret_cast<std::uintptr_t>(execution_context_.get())),
        " from context ",
        absl::Hex(reinterpret_cast<std::uintptr_t>(&execution_context)),
        " of state ", execution_context.state_)));
    execution_context.Await(std::move(*this));
  }

  bool IsReady() const { return shared_state_.IsAvailable(); }
  bool IsError() const { return shared_state_.IsError(); }

  const absl::Status& GetError() const { return shared_state_.GetError(); }

  ExecutionContext& execution_context() { return *execution_context_; }

 private:
  AsyncHandle(std::unique_ptr<ExecutionContext> execution_context,
              tsl::AsyncValueRef<tsl::Chain> shared_state)
      : execution_context_(std::move(execution_context)),
        shared_state_(std::move(shared_state)) {
    DCHECK(execution_context_);
    DCHECK(shared_state_);
  }

  std::unique_ptr<ExecutionContext> execution_context_;
  tsl::AsyncValueRef<tsl::Chain> shared_state_;
};

}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ASYNC_HANDLE_H_
