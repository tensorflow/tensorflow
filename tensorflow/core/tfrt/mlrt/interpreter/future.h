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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_FUTURE_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_FUTURE_H_

#include <atomic>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tfrt/concurrency/async_value.h"  // from @tf_runtime
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime

namespace mlrt {
namespace future_internal {

// The overloads of GetArgumentType() are used to get the argument type of a
// callable.
void GetArgumentType(void (*)());
template <typename F>
void GetArgumentType(void (F::*)());
template <typename F>
void GetArgumentType(void (F::*)() const);
template <typename Arg>
Arg GetArgumentType(void (*)(Arg));
template <typename F, typename Arg>
Arg GetArgumentType(void (F::*)(Arg));
template <typename F, typename Arg>
Arg GetArgumentType(void (F::*)(Arg) const);
template <typename F>
decltype(GetArgumentType(&F::operator())) GetArgumentType(F);

template <typename F>
using ArgumentType = decltype(GetArgumentType(std::declval<F>()));

template <typename T>
struct ArgTag {};

// The overloads of InvokeThen() are used to invoke different implementation
// according to `then`'s argument type.
template <typename F, typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE void InvokeThen(F&& then,
                                             tsl::AsyncValue* shared_state,
                                             ArgTag<T>) {
  auto& arg = shared_state->get<T>();
  if (shared_state->IsUnique()) {
    std::forward<F>(then)(std::move(arg));
  } else {
    std::forward<F>(then)(arg);
  }
}

template <typename F>
ABSL_ATTRIBUTE_ALWAYS_INLINE void InvokeThen(F&& then,
                                             tsl::AsyncValue* shared_state,
                                             ArgTag<absl::Status>) {
  if (shared_state->IsError()) {
    std::forward<F>(then)(shared_state->GetError());
  } else {
    std::forward<F>(then)(absl::OkStatus());
  }
}

template <typename F, typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE void InvokeThen(F&& then,
                                             tsl::AsyncValue* shared_state,
                                             ArgTag<absl::StatusOr<T>>) {
  if (shared_state->IsError()) {
    std::forward<F>(then)(shared_state->GetError());
  } else {
    InvokeThen(std::forward<F>(then), shared_state, ArgTag<T>());
  }
}

}  // namespace future_internal

struct Control {};

// mlrt::Future is similar to std::shared_future<T> but type-erased.
class Future {
 public:
  // Constructs a mlrt::Future directly from tsl::AsyncValue. This is used to
  // integrate MLRT with existing systems that uses AsyncValue directly. For new
  // use cases, creating mlrt::Future through mlrt::Promise is preferred.
  template <typename T>
  explicit Future(tsl::AsyncValueRef<T> async_value)
      : shared_state_(std::move(async_value)) {}

  Future(const Future& other) = default;
  Future& operator=(const Future& other) = default;
  Future(Future&& other) = default;
  Future& operator=(Future&& other) = default;

  explicit operator bool() const { return shared_state_ != nullptr; }

  bool IsReady() const {
    DCHECK(shared_state_);
    return shared_state_->IsAvailable();
  }

  bool IsError() const {
    DCHECK(shared_state_);
    return shared_state_->IsError();
  }

  template <typename T>
  const T& Get() const {
    DCHECK(shared_state_);
    return shared_state_->get<T>();
  }

  const absl::Status& GetError() const {
    DCHECK(shared_state_);
    return shared_state_->GetError();
  }

  // Then() enqueues a callback which will be called when the future is
  // fulfilled with either an error or a value.
  //
  // The following Then() overloads accept a callback with the following
  // signatures:
  //
  // 1) void(absl::StatusOr<T>)
  //    The argument can be either the error or the value.
  //
  // 2) void(absl::Status)
  //    The argument is the status of this future in ready state.
  //
  // 3) void(T)
  //    The argument is the fulfilled value. It is undefined behavior if there
  //    is an error.
  //
  // 4) void()
  //    There is no argument. The callback will be called whenever it is ready.

  template <typename F,
            typename Arg = std::decay_t<future_internal::ArgumentType<F>>>
  typename std::enable_if_t<!std::is_void_v<Arg>, void> Then(F then) && {
    DCHECK(shared_state_);
    auto* shared_state_ptr = shared_state_.get();
    shared_state_ptr->AndThen([shared_state = std::move(shared_state_),
                               then = std::move(then)]() mutable {
      future_internal::InvokeThen(std::move(then), shared_state.get(),
                                  future_internal::ArgTag<Arg>());
    });
  }

  template <typename F,
            typename Arg = std::decay_t<future_internal::ArgumentType<F>>>
  typename std::enable_if_t<std::is_void_v<Arg>, void> Then(F then) && {
    DCHECK(shared_state_);
    auto* shared_state_ptr = shared_state_.get();
    shared_state_ptr->AndThen(
        [shared_state = std::move(shared_state_),
         then = std::move(then)]() mutable { std::move(then)(); });
  }

  size_t UseCount() const {
    DCHECK(shared_state_);
    return shared_state_->NumRef();
  }

  // We don't need HandleError() method for Future because
  // AsyncHandle::HandleError() is enough for error handling for async
  // execution.

 private:
  friend class Promise;

  explicit Future(tsl::RCReference<tsl::AsyncValue> shared_state)
      : shared_state_(std::move(shared_state)) {}

  tsl::RCReference<tsl::AsyncValue> shared_state_;
};

// mlrt::Promise is similar to std::promise<T> but type-erased.
class Promise {
 public:
  template <typename T>
  static Promise Allocate() {
    return Promise(tsl::MakeUnconstructedAsyncValueRef<T>().ReleaseRCRef());
  }

  ~Promise() {
    DCHECK(!shared_state_ || shared_state_->IsAvailable())
        << "A non-empty promise must be fulfilled.";
  }

  Promise(const Promise&) = delete;
  Promise& operator=(const Promise&) = delete;
  Promise(Promise&&) = default;
  Promise& operator=(Promise&&) = default;

  Future GetFuture() const { return Future(shared_state_); }

  template <typename T, typename... Args>
  void Set(Args&&... args) && {
    DCHECK(shared_state_);

    auto shared_state = std::move(shared_state_);
    auto* shared_state_ptr = shared_state.get();

    // Since each waiter will hold a reference to the shared state, we can drop
    // the reference in mlrt::Promise::Set() in order to trigger passing by move
    // for the last waiter.
    if (!shared_state->IsUnique()) {
      shared_state.reset();
    }

    shared_state_ptr->emplace<T>(std::forward<Args>(args)...);
  }

  void SetError(absl::Status status) && {
    DCHECK(shared_state_);

    DCHECK(!status.ok());
    shared_state_->SetError(std::move(status));
    shared_state_.reset();
  }

  void HandleError(Value* arg) && {
    if (!shared_state_ || shared_state_->IsAvailable()) {
      // This is an empty promise or it is already fulfilled.
      return;
    }

    auto& execution_context = *arg->Get<ExecutionContext*>();
    DCHECK(!execution_context.status().ok());

    std::move(*this).SetError(execution_context.status());
  }

  explicit operator bool() const { return shared_state_ != nullptr; }

 private:
  explicit Promise(tsl::RCReference<tsl::AsyncValue> shared_state)
      : shared_state_(std::move(shared_state)) {}

  tsl::RCReference<tsl::AsyncValue> shared_state_;
};

namespace future_internal {

struct State {
  State(int size, mlrt::Promise promise)
      : count(size), promise(std::move(promise)) {}

  std::atomic<int> count;
  mlrt::Promise promise;

  absl::Mutex mu;
  absl::Status status;

  void SetError(absl::Status status) {
    absl::MutexLock lock(&mu);
    this->status = std::move(status);
  }

  // Returns true if it is the last consumer of the state. If this method
  // returns false, *this object might be destroyed anytime so the data can no
  // longer be accessed after it returns false.
  bool DecrementCount() {
    if (count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (status.ok()) {
        std::move(promise).Set<Control>(Control());
      } else {
        std::move(promise).SetError(std::move(status));
      }
      return true;
    }
    return false;
  }
};

}  // namespace future_internal

template <typename T, typename FutureLikeContainer, typename ResultRefContainer>
ABSL_ATTRIBUTE_ALWAYS_INLINE Future AwaitAll(FutureLikeContainer futures,
                                             ResultRefContainer results) {
  DCHECK(!futures.empty());

  auto promise = Promise::Allocate<Control>();
  auto await_all = promise.GetFuture();
  auto* state = new future_internal::State(futures.size(), std::move(promise));

  DCHECK_EQ(futures.size(), results.size());
  for (int i = 0; i < futures.size(); ++i) {
    auto& future = futures[i];
    std::move(future).Then(
        [state, result = &results[i]](absl::StatusOr<T> value) {
          if (value.ok()) {
            result->Set(std::move(*value));
          } else {
            state->SetError(std::move(value).status());
          }

          if (state->DecrementCount()) {
            delete state;
          }
        });
  }

  return await_all;
}

template <typename FutureLikeContainer>
ABSL_ATTRIBUTE_ALWAYS_INLINE Future AwaitAll(FutureLikeContainer futures) {
  DCHECK(!futures.empty());

  auto promise = Promise::Allocate<Control>();
  auto await_all = promise.GetFuture();
  auto* state = new future_internal::State(futures.size(), std::move(promise));

  for (int i = 0; i < futures.size(); ++i) {
    auto& future = futures[i];
    std::move(future).Then([state](absl::Status status) {
      if (!status.ok()) {
        state->SetError(std::move(status));
      }

      if (state->DecrementCount()) {
        delete state;
      }
    });
  }

  return await_all;
}

// TODO(chky): Implement type-safe version of Future and Promise.

}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_FUTURE_H_
