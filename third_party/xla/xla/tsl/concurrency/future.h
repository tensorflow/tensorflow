/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TSL_CONCURRENCY_FUTURE_H_
#define XLA_TSL_CONCURRENCY_FUTURE_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "absl/utility/utility.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {

// A Future<T> is a container for an asynchronous value of type T with value
// semantics. This is alternative to AsyncValueRef<T> which has a smart-pointer
// semantics.
//
// Key differences between Future<t> and AsyncValueRef<T>:
//
// 1. Reading and writing asynchronous value is split between Future<T> and
//    Promise<T>.
//
// 2. Promise<T> is a move-only type, and prevents accidentally setting the
//    value more than once through different copies of the Promise object.
//
// 3. Future<T> of move-only type `T` is also a move-only type, and it's
//    possible to safely move the value out of the future object once. This is
//    impossible to check at compile time with AsyncValueRef<T>.
//
// 4. Future<> is a stateless type that can be used to represent a Future that
//    doesn't have a value type, and is used to signal an event completion.
template <class T = void>
class Future;

// Returns a `Future` that will be successful if all `futures` complete
// successfully, or return a first encountered error.
Future<> JoinFutures(absl::Span<const Future<>> futures);

// Helpers for using Futures.
class FutureHelpers {
 public:
  // Keys that are returned by an implementation-specific handler when a client
  // starts to block on a promise.
  //
  // For now, contains a single UID that can be used to identify a TraceMe, but
  // made extensible to allow support for other profilers such as endoscope.
  struct ProfilingKeys {
    uint64_t traceme_context_id = -1;
  };

  // Signature of handler called by the Future class before it starts to
  // block a thread.
  using OnBlockStart = std::function<ProfilingKeys()>;

  // Signature of handler called by the Future class after it finishes
  // blocking a thread.
  using OnBlockEnd = std::function<void(ProfilingKeys)>;

  // Returns a Future<T> with optionally updated profiling handlers. If
  // profiling handlers are not provided, the original ones will be used.
  template <typename T>
  static Future<T> WithProfiling(Future<T> future,
                                 OnBlockStart on_block_start = nullptr,
                                 OnBlockEnd on_block_end = nullptr) {
    return Future<T>(std::move(future.promise_),
                     on_block_start ? std::move(on_block_start)
                                    : std::move(future.on_block_start_),
                     on_block_end ? std::move(on_block_end)
                                  : std::move(future.on_block_end_));
  }
};

namespace internal {

// A base class to conditionally disable copy constructor and assignment for a
// Future<T> (by default we always disable copy constructor when `T` is not
// copyable), which makes Future<T> an `std::unique_ptr`-like container for
// move-only types.
template <bool is_move_only>
class FutureMoveControl;

template <>
class FutureMoveControl</*is_move_only=*/true> {
 protected:
  FutureMoveControl() = default;

  FutureMoveControl(const FutureMoveControl&) = delete;
  FutureMoveControl& operator=(const FutureMoveControl&) = delete;

  FutureMoveControl(FutureMoveControl&&) = default;
  FutureMoveControl& operator=(FutureMoveControl&&) = default;
};

template <>
class FutureMoveControl</*is_move_only=*/false> {
 protected:
  FutureMoveControl() = default;

  FutureMoveControl(const FutureMoveControl&) = default;
  FutureMoveControl& operator=(const FutureMoveControl&) = default;

  FutureMoveControl(FutureMoveControl&&) = default;
  FutureMoveControl& operator=(FutureMoveControl&&) = default;
};

// A base class for a stateful future Future<T> and a stateless future Future<>.
// If `is_move_only` is true, Future derived from this class acts as a move-only
// type and the value can be passed to the caller only using move assignment
// (applied to Await and OnReady APIs).
template <typename T, bool is_move_only = !std::is_copy_constructible_v<T>>
class FutureBase : public FutureMoveControl<is_move_only> {
 protected:
  // A protected constructor that hides AsyncValueRef implementation detail
  // from the end users of Future and Promise. Must not be made public!
  FutureBase(tsl::AsyncValueRef<T> promise,
             FutureHelpers::OnBlockStart on_block_start,
             FutureHelpers::OnBlockEnd on_block_end)
      : promise_(std::move(promise)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)) {}

 public:
  FutureBase() = default;

  // Constructor for an already-available Future.
  //
  // Typically used to eagerly return error values when async work will not
  // be enqueued, e.g., due to invalid arguments.
  explicit FutureBase(T t, FutureHelpers::OnBlockStart on_block_start = nullptr,
                      FutureHelpers::OnBlockEnd on_block_end = nullptr)
      : FutureBase(tsl::MakeAvailableAsyncValueRef<T>(std::move(t)),
                   std::move(on_block_start), std::move(on_block_end)) {}

  bool IsValid() const { return promise_ != nullptr; }

  // Two functions exist to know whether the future is ready, to accommodate
  // the fact some backends (e.g. distributed ones) could take a non-trivial
  // time to check the state of a future.
  //
  // `IsReady()` is guaranteed to return true if the future became ready
  // before `IsReady()` was called. `IsReady()` will return immediately if a
  // call to `Await()` has already returned, or any callback passed to
  // `OnReady` has already been triggered. Otherwise IsReady() may block for
  // the duration of a network message on some backends.
  bool IsReady() const {
    CHECK(IsValid());
    return promise_.IsAvailable();
  }
  // `IsKnownReady()` is guaranteed to return immediately. `IsKnownReady()` will
  // always return true if a call to `Await()` has already returned, or any
  // callback passed to `OnReady` has already been triggered. Otherwise,
  // `IsKnownReady()` may return false in some cases in which the future was
  // ready before `IsKnownReady()` was called.
  bool IsKnownReady() const {
    CHECK(IsValid());
    return promise_.IsAvailable();
  }

  explicit operator bool() const { return static_cast<bool>(promise_); }

  // Returns a pointer to the underlying AsyncValue that can be used to
  // track completion of a future. It is undefined behavior to access the
  // value stored in the AsyncValue.
  tsl::AsyncValue* async_value() const { return promise_.GetAsyncValue(); }

 protected:
  static constexpr bool IsMoveOnly() { return is_move_only; }

  // Future<T>::Promise provides a facility to store a value or an error
  // that is later acquired asynchronously via a Future<T> constructed from
  // the promise object. Note that the promise object is meant to be used only
  // once (set value or error).
  class Promise {
   public:
    Promise() = default;

    Promise(Promise&& other) = default;
    Promise& operator=(Promise&& other) = default;

    ~Promise() {
      if (promise_ && !IsUniqueReference() && promise_.IsUnavailable()) {
        // At this point, we know that the underlying AsyncValueRef will
        // otherwise not fulfilled ever because `Promise` is move-only.
        promise_.emplace(
            absl::InternalError("Promise destroyed without being set"));
      }
    }

    explicit operator bool() const { return static_cast<bool>(promise_); }

    // Returns if this promise is the unique reference to the underlying value.
    // That is, this method returns true only if all of the following conditions
    // are satisfied:
    //
    // - The promise is the only reference to the underlying value, i.e., there
    //   are no other promises or futures associated with this value.
    // - There are no OnReady callbacks registered to this promise.
    //
    // This may be used by the caller of `Set()` to short-circuit the work to
    // fulfill the promise if no one will ever consume the value. Even in that
    // case, consider fulfilling the promise with an error (e.g., `CANCELLED`)
    // instead of dropping the promise without fulfilling it in order to make
    // debugging easier. Also, be aware that the current promise may still be
    // used to mint a future.
    bool IsUniqueReference() const {
      CHECK(promise_) << "Promise must wrap an async value";
      return promise_.IsUnique() && !promise_.HasWaiter();
    }

   protected:
    explicit Promise(tsl::AsyncValueRef<T> promise)
        : promise_(std::move(promise)) {}

    template <typename... Args>
    void emplace(Args&&... args) const {
      CHECK(promise_) << "Promise must wrap an async value";
      CHECK(promise_.IsUnavailable())
          << "Promise must not be fulfilled more than once";
      promise_.template emplace<Args...>(std::forward<Args>(args)...);
    }

    // Takes a reference to the underlying AsyncValueRef container.
    tsl::AsyncValueRef<T> ref() const { return promise_; }

   private:
    tsl::AsyncValueRef<T> promise_;
  };

  class ProfilingCleanup {
   public:
    ProfilingCleanup(const FutureBase* parent,
                     FutureHelpers::ProfilingKeys keys)
        : parent_(parent), keys_(std::move(keys)) {}
    ~ProfilingCleanup() {
      if (parent_ && parent_->on_block_end_) {
        parent_->on_block_end_(std::move(keys_));
      }
    }
    ProfilingCleanup(const ProfilingCleanup& other) = delete;
    ProfilingCleanup(ProfilingCleanup&& other) = delete;

   private:
    const FutureBase* parent_;
    FutureHelpers::ProfilingKeys keys_;
  };

  ProfilingCleanup OnBlockStartScope() const {
    return ProfilingCleanup(this, on_block_start_
                                      ? on_block_start_()
                                      : FutureHelpers::ProfilingKeys());
  }

  // Calls block_until_ready_fn to wait until the underlying AsyncValue is
  // concrete. block_until_ready_fn should be equivalent to
  // tsl::BlockUntilReady.
  template <typename Fn>
  void BlockUntilReady(Fn&& block_until_ready_fn) const {
    CHECK(IsValid());
    if (!promise_.IsAvailable()) {
      ProfilingCleanup scope = OnBlockStartScope();
      block_until_ready_fn(promise_.GetAsyncValue());
    }
    DCHECK(promise_.IsConcrete());
  }

  // Blocks the calling thread until the future is ready, then returns the
  // final value.
  const T& Await() const& {
    BlockUntilReady(
        static_cast<void (*)(tsl::AsyncValue*)>(tsl::BlockUntilReady));
    return *promise_;
  }

  // Blocks the calling thread until the future is ready, then returns the
  // final value.
  std::conditional_t<is_move_only, T, const T&> Await() && {
    BlockUntilReady(
        static_cast<void (*)(tsl::AsyncValue*)>(tsl::BlockUntilReady));

    if constexpr (is_move_only) {
      return std::move(*promise_);
    } else {
      // We can't move from the promise to the caller because for copyable
      // futures we can have multiple copies of the Future sharing the
      // same underlying promise object.
      return *promise_;
    }
  }

  // Returns a Future<> that becomes ready when *this is ready. If *this
  // completes with an error, the returned future will also be an error.
  //
  // This function defined out of line as it requires Future<> definition.
  Future<> GetReadyFuture() const;

  // Registers callback to be called once the promise is ready, with the final
  // value.
  //
  // callback may be called on an internal system thread or the calling thread.
  // The client should avoid any potentially re-entrant API calls within the
  // callback, for example by using the callback to enqueue work on a
  // client-owned threadpool.
  template <typename F,
            std::enable_if_t<!is_move_only &&
                             std::is_invocable_v<F, const T&>>* = nullptr>
  void OnReady(F&& f) const& {
    CHECK(IsValid());
    promise_.AndThen(
        [promise = promise_.AsPtr(), f = std::forward<F>(f)]() mutable {
          DCHECK(promise.IsConcrete());
          f(*promise);
        });
  }

  // Registers callback to be called once the promise is ready, with the final
  // value.
  //
  // callback may be called on an internal system thread or the calling thread.
  // The client should avoid any potentially re-entrant API calls within the
  // callback, for example by using the callback to enqueue work on a
  // client-owned threadpool.
  template <typename F,
            std::enable_if_t<is_move_only ? std::is_invocable_v<F, T>
                                          : std::is_invocable_v<F, const T&>>* =
                nullptr>
  void OnReady(F&& f) && {
    CHECK(IsValid());
    promise_.AndThen(
        [promise = promise_.AsPtr(), f = std::forward<F>(f)]() mutable {
          DCHECK(promise.IsConcrete());
          if constexpr (is_move_only) {
            f(std::move(*promise));
          } else {
            // We can't move from the promise to the caller because for copyable
            // futures we can have multiple copies of the Future sharing the
            // same underlying promise object.
            f(*promise);
          }
        });
    // Reset the promise to make the moved-from future "empty".
    promise_.reset();
  }

 protected:
  // Returns a placeholder error that can be used when short-circuiting promises
  // with no other references.
  static absl::Status AbortedError() {
    return absl::AbortedError(
        "Fulfilling the promise with an aborted error since the value is no "
        "longer referenced by any futures or OnReady callbacks; if this error "
        "is exposed to any future, that indicates a bug");
  }

 private:
  friend class tsl::FutureHelpers;

  tsl::AsyncValueRef<T> promise_;

  // Function that is called before a thread starts blocking on the promise.
  FutureHelpers::OnBlockStart on_block_start_;
  // Function that is called after a thread finishes blocking on the promise.
  FutureHelpers::OnBlockEnd on_block_end_;
};

}  // namespace internal

// Future<T> is a simple future that is returned by  APIs that enqueue
// asynchronous work, reporting a value of type T when the work is complete.
//
// Future can be used by the client to wait for work to complete, either via
// a blocking call or a callback.
//
// The implementation wraps a AsyncValueRef<T>, but in contrast to AsyncValueRef
// which has a smart-pointer semantics, future has a value semantics, i.e.
// future of a move-only type also is a move-only type. You can think of a
// move-only future as a box to pass a value of type `T` between asynchronous
// producer/consumer: you can open the box once to put the value into it and you
// can open the box only once to take the value out of it. For copyable types
// Future<T> is a copyable type, although all copies share the same underlying
// async value.
template <class T>
class Future : public internal::FutureBase<absl::StatusOr<T>> {
  using Base = internal::FutureBase<absl::StatusOr<T>>;

  static constexpr bool is_move_only = Base::IsMoveOnly();  // NOLINT

  template <typename U>
  static constexpr bool is_status_or =  // NOLINT
      tsl::internal::is_status_or_v<U>;

  static_assert(!std::is_same_v<T, absl::Status>,
                "Use Future<> specialization for stateless futures");

  static_assert(
      !internal::IsStatusOr<T>::value,
      "Future<T> already has an implicit absl::StatusOr<T> semantics");

 public:
  Future() = default;

  // Constructs an immediately available future with the given value.
  explicit Future(absl::StatusOr<T> value) : Base(std::move(value)) {}

  // Constructs and immediately available future from the given value.
  template <typename U,
            std::enable_if_t<std::is_constructible_v<T, U>>* = nullptr>
  explicit Future(U value) : Base(std::forward<U>(value)) {}

  class Promise : public Base::Promise {
   public:
    using Base::Promise::Promise;

    // Sets the value of the promise. Must be called at most once.
    //
    // After Set is called, value will be delivered to waiters on the Future
    // constructed from a promise, via blocking or callbacks.
    void Set(absl::StatusOr<T> value) {
      Base::Promise::emplace(std::move(value));
    }

    // A helper function to convert move-only Promise to shared_ptr, which is
    // useful when the promise has to be captured by a std::function.
    std::shared_ptr<Promise> ToShared() && {
      return std::make_shared<Promise>(std::move(*this));
    }

    // Returns a future associated with the promise. We use a trick we an extra
    // template parameter to disable converting promise to future for move-only
    // types, as it is illegal to create multiple move-only futures sharing the
    // underlying async value storage. For move-only types, the only way to
    // create a future is to call `MakePromise`.
    template <typename U = void,
              std::enable_if_t<!is_move_only && std::is_void_v<U>>* = nullptr>
    Future<T> future(FutureHelpers::OnBlockStart on_block_start = nullptr,
                     FutureHelpers::OnBlockEnd on_block_end = nullptr) const {
      return Future<T>(*this, std::move(on_block_start),
                       std::move(on_block_end));
    }

   private:
    template <typename>
    friend class Future;
  };

  // Returns a pair of connected Promise and Future<T>. Setting the returned
  // promise will fulfill the connected future.
  //
  // - on_block_start is called before Await starts to block.
  // - on_block_end is called after Await finishes blocking.
  static std::pair<Promise, Future<T>> MakePromise(
      FutureHelpers::OnBlockStart on_block_start = nullptr,
      FutureHelpers::OnBlockEnd on_block_end = nullptr) {
    Promise promise(tsl::MakeUnconstructedAsyncValueRef<absl::StatusOr<T>>());
    Future<T> future(promise, std::move(on_block_start),
                     std::move(on_block_end));
    return std::make_pair(std::move(promise), std::move(future));
  }

  // Returns a future that is constructed from the result of invoking functor
  // `f` on the given `executor`.
  template <typename F, typename R = std::invoke_result_t<F>,
            std::enable_if_t<std::is_constructible_v<absl::StatusOr<T>, R>>* =
                nullptr>
  static Future<T> MakeOn(Executor& executor, F&& f) {
    auto [promise, future] = MakePromise();
    executor.Execute([promise = std::move(promise),
                      f = std::forward<F>(f)]() mutable { promise.Set(f()); });
    return std::move(future);
  }

  using Base::Await;
  using Base::GetReadyFuture;
  using Base::OnReady;

  // Returns an Future<R> that is constructed from the result of invoking
  // functor `f` with *this value. If *this completes with an error, returned
  // future will also be an error.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage:
  //
  // future.Map<R>([](const T& value) -> U {
  //   return U(value); // R must be constructible from U
  // })
  //
  template <typename R, typename F,
            typename U = std::invoke_result_t<F, const T&>,
            std::enable_if_t<!is_move_only && std::is_constructible_v<R, U>>* =
                nullptr>
  Future<R> Map(F&& f) const& {
    auto [promise, future] = Future<R>::MakePromise();

    using Value = const absl::StatusOr<T>&;
    OnReady([promise = std::move(promise),
             f = std::forward<F>(f)](Value value) mutable {
      if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
        promise.Set(Base::AbortedError());
      } else if (ABSL_PREDICT_TRUE(value.ok())) {
        promise.emplace(absl::in_place_t{}, f(*value));
      } else {
        promise.Set(value.status());
      }
    });

    return std::move(future);
  }

  // Returns an Future<R> that is constructed from the result of invoking
  // functor `f` with *this value. If *this completes with an error, returned
  // future will also be an error.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage: move-only type T passed by value
  //
  // std::move(future).Map<R>([](T value) -> U {
  //   return U(std::move(value)); // R must be constructible from U
  // })
  //
  template <typename R, typename F,
            typename U = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>,
            std::enable_if_t<std::is_constructible_v<R, U>>* = nullptr>
  Future<R> Map(F&& f) && {
    auto [promise, future] = Future<R>::MakePromise();

    using Value = std::conditional_t<is_move_only, absl::StatusOr<T>,
                                     const absl::StatusOr<T>&>;
    std::move(*this).OnReady([promise = std::move(promise),
                              f = std::forward<F>(f)](Value value) mutable {
      if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
        promise.Set(Base::AbortedError());
      } else if (ABSL_PREDICT_TRUE(value.ok())) {
        if constexpr (is_move_only) {
          promise.emplace(absl::in_place_t{}, f(std::move(*value)));
        } else {
          promise.emplace(absl::in_place_t{}, f(*value));
        }
      } else {
        promise.Set(value.status());
      }
    });

    return std::move(future);
  }

  // Returns an Future<R> that is constructed from the result of invoking
  // functor `f` with *this value. If *this completes with an error, returned
  // future will also be an error. Functor `f` must return a value of type
  // absl::StatusOr<U> where R is constructible from U. Returned absl::StatusOr
  // is automatically unwrapped and returned as a future payload.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage:
  //
  // future.TryMap<R>([](const T& value) -> absl::StatusOr<U> {
  //   return U(value); // R must be constructible from U
  // })
  //
  template <
      typename R, typename F, typename U = std::invoke_result_t<F, const T&>,
      std::enable_if_t<!is_move_only && is_status_or<U> &&
                       std::is_constructible_v<R, typename U::value_type>>* =
          nullptr>
  Future<R> TryMap(F&& f) const& {
    auto [promise, future] = Future<R>::MakePromise();

    using Value = const absl::StatusOr<T>&;
    OnReady([promise = std::move(promise),
             f = std::forward<F>(f)](Value value) mutable {
      if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
        promise.Set(Base::AbortedError());
      } else if (ABSL_PREDICT_TRUE(value.ok())) {
        auto result = f(*value);
        if (ABSL_PREDICT_TRUE(result.ok())) {
          promise.emplace(absl::in_place_t{}, *std::move(result));
        } else {
          promise.Set(std::move(result).status());
        }
      } else {
        promise.Set(value.status());
      }
    });

    return std::move(future);
  }

  // Returns an Future<R> that is constructed from the result of invoking
  // functor `f` with *this value. If *this completes with an error, returned
  // future will also be an error. Functor `f` must return a value of type
  // absl::StatusOr<U> where R is constructible from U. Returned absl::StatusOr
  // is automatically unwrapped and returned as a future payload.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage: move-only type T passed by rvalue
  //
  // future.TryMap<R>([](const T& value) -> absl::StatusOr<U> {
  //   return U(value); // R must be constructible from U
  // })
  //
  template <typename R, typename F,
            typename U = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>,
            std::enable_if_t<
                is_status_or<U> &&
                std::is_constructible_v<R, typename U::value_type>>* = nullptr>
  Future<R> TryMap(F&& f) && {
    auto [promise, future] = Future<R>::MakePromise();

    using Value = std::conditional_t<is_move_only, absl::StatusOr<T>,
                                     const absl::StatusOr<T>&>;
    std::move(*this).OnReady([promise = std::move(promise),
                              f = std::forward<F>(f)](Value value) mutable {
      if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
        promise.Set(Base::AbortedError());
      } else if (ABSL_PREDICT_TRUE(value.ok())) {
        auto result = [&] {
          if constexpr (is_move_only) {
            return f(std::move(*value));
          } else {
            return f(*value);
          }
        }();
        if (ABSL_PREDICT_TRUE(result.ok())) {
          promise.emplace(absl::in_place_t{}, *std::move(result));
        } else {
          promise.Set(std::move(result).status());
        }
      } else {
        promise.Set(value.status());
      }
    });

    return std::move(future);
  }

  // A `Map` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<F, const T&>>
  Future<R> Map(F&& f) const& {
    return Map<R>(std::forward<F>(f));
  }

  // A `Map` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<
                            F, std::conditional_t<is_move_only, T, const T&>>>
  Future<R> Map(F&& f) && {
    return std::move(*this).template Map<R>(std::forward<F>(f));
  }

  // A `TryMap` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<F, const T&>,
            std::enable_if_t<is_status_or<R>>* = nullptr>
  Future<typename R::value_type> TryMap(F&& f) const& {
    return TryMap<typename R::value_type>(std::forward<F>(f));
  }

  // A `TryMap` overload that automatically infers the type of result from `f`.
  template <typename F,
            typename R = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>,
            std::enable_if_t<is_status_or<R>>* = nullptr>
  Future<typename R::value_type> TryMap(F&& f) && {
    return std::move(*this).template TryMap<typename R::value_type>(
        std::forward<F>(f));
  }

 private:
  friend class FutureHelpers;

  // Bring FutureBase constructors in scope.
  using Base::Base;

  // Constructor for unavailable future that will be fulfilled later via the
  // promise object.
  //
  // - on_block_start is called before Await starts to block.
  // - on_block_end is called after Await finishes blocking.
  Future(const Promise& promise, FutureHelpers::OnBlockStart on_block_start,
         FutureHelpers::OnBlockEnd on_block_end)
      : Base(promise.ref(), std::move(on_block_start),
             std::move(on_block_end)) {}
};

// Future<void> specialization for communicating stateless events.
//
// See Future<T> documentation above for more details.
template <>
class Future<void> : public internal::FutureBase<absl::Status> {
  using Base = internal::FutureBase<absl::Status>;

  template <typename U>
  static constexpr bool is_status_or =  // NOLINT
      tsl::internal::is_status_or_v<U>;

 public:
  Future() = default;

  // Constructor for a future that is immediately ready with a given status.
  // For futures that are immediately ready with OK status, we use a global non
  // reference-counted async value that avoids heap allocation and reference
  // counting operations on a hot path.
  explicit Future(absl::Status status)
      : Base(ABSL_PREDICT_TRUE(status.ok())
                 ? ready_promise_->AsRef()
                 : tsl::MakeAvailableAsyncValueRef<absl::Status>(
                       std::move(status)),
             /*on_block_start=*/nullptr, /*on_block_end=*/nullptr) {}

  class Promise : public Base::Promise {
   public:
    using Base::Promise::Promise;

    // Sets the promise completed with a given status. Must be called at most
    // once.
    //
    // After Set is called, completion event will be delivered to waiters on the
    // Future constructed from a promise, via blocking or callbacks.
    void Set(absl::Status status = absl::OkStatus()) {
      Base::Promise::emplace(std::move(status));
    }

    // A helper function to convert move-only Promise to shared_ptr, which is
    // useful when the promise has to be captured by a std::function.
    std::shared_ptr<Promise> ToShared() && {
      return std::make_shared<Promise>(std::move(*this));
    }

    // Returns a future associated with the promise.
    Future<> future(FutureHelpers::OnBlockStart on_block_start = nullptr,
                    FutureHelpers::OnBlockEnd on_block_end = nullptr) const {
      return Future<>(*this, std::move(on_block_start),
                      std::move(on_block_end));
    }

   private:
    friend class Future<void>;
  };

  // Returns a pair of connected Promise and Future<>. Setting the returned
  // promise will fulfill the connected future.
  static std::pair<Promise, Future<>> MakePromise(
      FutureHelpers::OnBlockStart on_block_start = nullptr,
      FutureHelpers::OnBlockEnd on_block_end = nullptr) {
    Promise promise(tsl::MakeUnconstructedAsyncValueRef<absl::Status>());
    Future<> future(promise, std::move(on_block_start),
                    std::move(on_block_end));
    return std::make_pair(std::move(promise), std::move(future));
  }

  // Returns a future that is constructed from the result of invoking functor
  // `f` on the given `executor`.
  template <typename F, typename R = std::invoke_result_t<F>,
            std::enable_if_t<std::is_same_v<R, absl::Status>>* = nullptr>
  static Future<> MakeOn(Executor& executor, F&& f) {
    auto [promise, future] = MakePromise();
    executor.Execute([promise = std::move(promise),
                      f = std::forward<F>(f)]() mutable { promise.Set(f()); });
    return std::move(future);
  }

  using Base::Await;
  using Base::BlockUntilReady;
  using Base::OnReady;

  // Returns an Future<R> that is constructed from the result of invoking
  // functor `f`. If *this completes with an error, returned future will also be
  // an error.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage:
  //
  // future.Map<R>([]() -> U {
  //   return U(value); // R must be constructible from U
  // })
  //
  template <typename R, typename F, typename U = std::invoke_result_t<F>>
  Future<R> Map(F&& f) {
    auto [promise, future] = Future<R>::MakePromise();

    OnReady([promise = std::move(promise),
             f = std::forward<F>(f)](absl::Status status) mutable {
      if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
        promise.Set(Base::AbortedError());
      } else if (ABSL_PREDICT_TRUE(status.ok())) {
        promise.emplace(absl::in_place_t{}, f());
      } else {
        promise.Set(std::move(status));
      }
    });

    return std::move(future);
  }

  // Returns an Future<R> that is constructed from the result of invoking
  // functor `f`. If *this completes with an error, returned future will also be
  // an error. Functor `f` must return a value of type absl::StatusOr<U> where R
  // is constructible from U. Returned absl::StatusOr is automatically unwrapped
  // and returned as a future payload.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage:
  //
  // future.TryMap<R>([]() -> absl::StatusOr<U> {
  //   return U(value); // R must be constructible from U
  // })
  //
  template <typename R, typename F, typename U = std::invoke_result_t<F>,
            std::enable_if_t<
                is_status_or<U> &&
                std::is_constructible_v<R, typename U::value_type>>* = nullptr>
  Future<R> TryMap(F&& f) {
    auto [promise, future] = Future<R>::MakePromise();

    OnReady([promise = std::move(promise),
             f = std::forward<F>(f)](absl::Status status) mutable {
      if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
        promise.Set(Base::AbortedError());
      } else if (ABSL_PREDICT_TRUE(status.ok())) {
        auto result = f();
        if (ABSL_PREDICT_TRUE(result.ok())) {
          promise.emplace(absl::in_place_t{}, *std::move(result));
        } else {
          promise.Set(std::move(result).status());
        }
      } else {
        promise.Set(std::move(status));
      }
    });

    return std::move(future);
  }

  // A `Map` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<F>>
  Future<R> Map(F&& f) {
    return Map<R>(std::forward<F>(f));
  }

  // A `TryMap` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<F>,
            std::enable_if_t<is_status_or<R>>* = nullptr>
  Future<typename R::value_type> TryMap(F&& f) {
    return TryMap<typename R::value_type>(std::forward<F>(f));
  }

  // Returns an Future<R> that is constructed from the given value. If *this
  // completes with an error, returned future will also be an error.
  //
  // Note: The implementation may choose to not run `f` if it can infer that the
  // returned future will never be used. Do not use this method if `f` has a
  // side effect that must always be executed when the future becomes ready.
  //
  // Sample usage: make buffer available when future is ready
  //
  // std::unique_ptr<Buffer> buffer = ...;
  // future.MapTo<R>(std::move(buffer));
  template <typename R>
  Future<absl::remove_cvref_t<R>> MapTo(R&& value) {
    return Map<absl::remove_cvref_t<R>>(
        [value = std::forward<R>(value)]() mutable {
          return std::move(value);
        });
  }

 private:
  friend class FutureHelpers;

  // A promise that is immediately ready with OK status. Async value allocated
  // in the static storage and is not reference-counted.
  static absl::NoDestructor<tsl::AsyncValueOwningRef<absl::Status>>
      ready_promise_;

  // Bring FutureBase constructors in scope.
  using Base::Base;

  // Constructor for unavailable future that will be fulfilled later via the
  // promise object.
  //
  // - on_block_start is called before Await starts to block.
  // - on_block_end is called after Await finishes blocking.
  Future(const Promise& promise, FutureHelpers::OnBlockStart on_block_start,
         FutureHelpers::OnBlockEnd on_block_end)
      : Base(promise.ref(), std::move(on_block_start),
             std::move(on_block_end)) {}
};

// Bring Promise implementation into the tsl namespace.
template <typename T = void>
using Promise = typename Future<T>::Promise;  // NOLINT

//===----------------------------------------------------------------------===//
// internal::FutureBase<T> implementation.
//===----------------------------------------------------------------------===//

namespace internal {

template <typename T, bool is_move_only>
Future<> FutureBase<T, is_move_only>::GetReadyFuture() const {
  auto [promise, future] = Future<>::MakePromise();
  promise_.AndThen(
      [self = promise_.AsPtr(), promise = std::move(promise)]() mutable {
        if constexpr (std::is_same_v<T, absl::Status>) {
          promise.Set(*self);
        } else {
          promise.Set(self->status());
        }
      });
  return std::move(future);
}

}  // namespace internal
}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_FUTURE_H_
