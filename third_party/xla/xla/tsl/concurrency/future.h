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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/bind_front.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/ref_count.h"
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

// Promise<T> provides a facility to store a value or an error that is later
// acquired asynchronously via a Future<T> constructed from the promise object.
// Note that the promise object is meant to be used only once (set value or
// error).
template <class T = void>
class Promise;

namespace internal {
// Type predicate to check that type is a future.
template <typename T>
struct IsFuture : std::false_type {};
template <typename T>
struct IsFuture<Future<T>> : std::true_type {};
}  // namespace internal

// Returns a `Future` that will be successful if all `futures` complete
// successfully, or return a first encountered error.
Future<> JoinFutures(absl::Span<const Future<>> futures);

// Returns a `Future` that will be successful if all `futures` complete
// successfully, or return a first encountered error. Copies values from
// completed futures into the result vector.
template <typename T, std::enable_if_t<!std::is_void_v<T>>* = nullptr>
Future<std::vector<T>> JoinFutures(absl::Span<const Future<T>> futures);

// Returns a `Future` that will be successful if all `futures` complete
// successfully, or return a first encountered error. Moves values from
// completed futures into the result vector and leaves `futures` in move-from
// state (for copyable `T` it still incurs a copy overhead, see `OnReady`
// documentation for details).
template <typename T, std::enable_if_t<!std::is_void_v<T>>* = nullptr>
Future<std::vector<T>> JoinFutures(absl::Span<Future<T>> futures);

// Returns a `Future` that will be successful if all `futures` complete
// successfully, or return a first encountered error. Return type is
// automatically derived from the passed futures.
//
// Example:
//
//   Future<std::string> f0 = ...;
//   Future<std::string> f1 = ...;
//   Future<std::tuple<std::string, int32_t> joined = JoinFutures(f0, f1);
//
//
// Example with custom joined type:
//
//   struct TwoInts {
//     int32_t a;
//     int32_t b;
//   };
//
//   Future<int32_t> f0 = ...;
//   Future<int32_t> f1 = ...;
//   Future<TwoInts> joined = JoinFutures<TwoInts>(f0, f1);
//
// If custom result type for `JoinFutures` is not defined (is void by default),
// then the result type will be inferred as `std::tuple`. Otherwise the result
// value of type `R` will be constructed from expanded tuple values.
template <typename R = void, typename... Futures,
          std::enable_if_t<std::conjunction_v<
              internal::IsFuture<std::decay_t<Futures>>...>>* = nullptr>
auto JoinFutures(Futures&&... futures);

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
  template <int&... ExplicitParameterBarrier, typename T>
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

// A template helper to deduce the `Future` type from the `FutureBase` type.
// clang-format off
template <typename T>
struct FutureType;
template <>
struct FutureType<absl::Status>      { using type = void; };
template <typename T>
struct FutureType<absl::StatusOr<T>> { using type = T; };
// clang-format on

template <typename T>
using future_type_t = typename FutureType<T>::type;  // NOLINT

// Detect if type `T` is move only.
template <typename T>
struct IsMoveOnly {
  static constexpr bool value =
      std::is_move_constructible_v<T> && !std::is_copy_constructible_v<T>;
};

// STL containers do not correctly report `std::is_copy_constructible_v` for
// move-only value type.
template <typename T>
struct IsMoveOnly<std::vector<T>> {
  static constexpr bool value = IsMoveOnly<T>::value;
};

// Unwrap `absl::StatusOr` container.
template <typename T>
struct IsMoveOnly<absl::StatusOr<T>> {
  static constexpr bool value = IsMoveOnly<T>::value;
};

// A base class for a stateful future Future<T> and a stateless future Future<>.
// If `is_move_only` is true, Future derived from this class acts as a move-only
// type and the value can be passed to the caller only using move assignment
// (applied to Await and OnReady APIs).
template <typename T, bool is_move_only = IsMoveOnly<T>::value>
class FutureBase : public FutureMoveControl<is_move_only> {
  static_assert(internal::is_status_v<T> || internal::is_status_or_v<T>,
                "Future value type must be absl::Status or absl::StatusOr");

  // A type predicate to check if `F` is a valid `OnReady` callback.
  template <typename F, bool rvalue = false>
  using OnReadyFunctor = std::enable_if_t<std::is_invocable_v<
      F, std::conditional_t<rvalue && is_move_only, T, const T&>>>;

 protected:
  FutureBase() = default;

  // A protected constructor that hides AsyncValueRef implementation detail
  // from the end users of Future and Promise. Must not be made public!
  FutureBase(tsl::AsyncValueRef<T> promise,
             FutureHelpers::OnBlockStart on_block_start,
             FutureHelpers::OnBlockEnd on_block_end)
      : promise_(std::move(promise)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)) {}

  // Constructor for an already-available Future.
  template <int&... ExplicitParameterBarrier, typename U,
            std::enable_if_t<std::is_constructible_v<T, U> ||
                             std::is_same_v<T, U>>* = nullptr>
  explicit FutureBase(U&& value)
      : FutureBase(tsl::MakeAvailableAsyncValueRef<T>(std::forward<U>(value)),
                   /*on_block_start=*/nullptr, /*on_block_end=*/nullptr) {}

 public:
  using value_type = T;

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
  template <int&... ExplicitParameterBarrier, typename Fn>
  void BlockUntilReady(Fn&& block_until_ready_fn) const {
    CHECK(IsValid());
    if (!promise_.IsAvailable()) {
      ProfilingCleanup scope = OnBlockStartScope();
      block_until_ready_fn(promise_.GetAsyncValue());
    }
    DCHECK(promise_.IsAvailable());
  }

  // Blocks the calling thread until the future is ready, then returns the
  // final value.
  [[nodiscard]] const T& Await() const& {
    BlockUntilReady(
        static_cast<void (*)(tsl::AsyncValue*)>(tsl::BlockUntilReady));
    return *promise_;
  }

  // Blocks the calling thread until the future is ready, then returns the
  // final value.
  [[nodiscard]] std::conditional_t<is_move_only, T, const T&> Await() && {
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

  // Returns a detached `Future<T>` that by default will execute all `OnReady`
  // callbacks (and `Map` functors) on the given `executor`.
  //
  // When future value is set via the connected promise, all callbacks attached
  // to the future will be executed on a thread that sets the promise value.
  // This might lead to unexpectedly running expensive callbacks on a thread
  // that is not intended for that, i.e. if a promise is set by a non-blocking
  // thread that handles IO events, running expensive computation might lead to
  // overall performance degradation.
  //
  // Detached future guarantees that all pending callbacks will be executed on
  // the specified executor. If the future is ready when `OnReady` or `Map` is
  // called, then the callback will be executed immediately in the caller
  // thread. Users can explicitly override executor by using `OnReady` and `Map`
  // overloads that accept another executor instance.
  //
  // We use a trick we an extra template parameter to disable const& overload
  // when T is move-only, as we don't want to allow to create multiple futures
  // sharing the same async value promise.
  template <int&... ExplicitParameterBarrier, typename U = void,
            std::enable_if_t<!is_move_only && std::is_void_v<U>>* = nullptr>
  [[nodiscard]] Future<future_type_t<T>> Detach(Executor& executor) const&;
  [[nodiscard]] Future<future_type_t<T>> Detach(Executor& executor) &&;

  // Returns a Future<> that becomes ready when *this is ready. If *this
  // completes with an error, the returned future will also be an error.
  //
  // This function defined out of line as it requires Future<> definition.
  [[nodiscard]] Future<> GetReadyFuture() const;

  // Registers callback to be called once the promise is ready, with the final
  // value. Callback will be invoked on a thread that sets the promise value,
  // or in the caller thread if the future is already available.
  template <int&... ExplicitParameterBarrier, typename F,
            OnReadyFunctor<F>* = nullptr>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void OnReady(F&& f) const& {
    CHECK(IsValid());
    promise_.AndThen(AndThen(std::forward<F>(f)));
  }

  // Registers callback to be called once the promise is ready, with the final
  // value. Callback will be invoked on a user-specified executor.
  template <int&... ExplicitParameterBarrier, typename F,
            OnReadyFunctor<F>* = nullptr>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void OnReady(Executor& executor, F&& f) const& {
    CHECK(IsValid());
    promise_.AndThen(executor, AndThen(std::forward<F>(f)));
  }

  // Registers callback to be called once the promise is ready, with the final
  // value. Callback will be invoked on a thread that sets the promise value,
  // or in the caller thread if the future is already available.
  template <int&... ExplicitParameterBarrier, typename F,
            OnReadyFunctor<F, true>* = nullptr>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void OnReady(F&& f) && {
    CHECK(IsValid());
    promise_.AndThen(std::move(*this).AndThen(std::forward<F>(f)));
    promise_.reset();
  }

  // Registers callback to be called once the promise is ready, with the final
  // value. Callback will be invoked on a user-specified executor.
  template <int&... ExplicitParameterBarrier, typename F,
            OnReadyFunctor<F, true>* = nullptr>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void OnReady(Executor& executor, F&& f) && {
    CHECK(IsValid());
    promise_.AndThen(executor, std::move(*this).AndThen(std::forward<F>(f)));
    promise_.reset();
  }

  // Returns a placeholder error that can be used when short-circuiting promises
  // with no other references.
  static absl::Status AbortedError() {
    return absl::AbortedError(
        "Fulfilling the promise with an aborted error since the value is no "
        "longer referenced by any futures or OnReady callbacks; if this error "
        "is exposed to any future, that indicates a bug");
  }

  // Returns a non-owning pointer to the underlying AsyncValue container.
  AsyncValuePtr<T> promise() const { return promise_.AsPtr(); }

 private:
  friend class ::tsl::FutureHelpers;

  // Wraps a callback into a functor compatible with AsyncValue::AndThen.
  template <typename F>
  auto AndThen(F&& f) const& {
    return [ptr = promise_.AsPtr(), f = std::forward<F>(f)]() mutable {
      std::move(f)(*ptr);
    };
  }

  // Wraps a callback into a functor compatible with AsyncValue::AndThen.
  template <typename F>
  auto AndThen(F&& f) && {
    return [ptr = promise_.AsPtr(), f = std::forward<F>(f)]() mutable {
      if constexpr (is_move_only) {
        std::move(f)(std::move(*ptr));
      } else {
        // We can't move from the promise to the caller because for copyable
        // futures we can have multiple copies of the Future sharing the
        // same underlying promise object.
        std::move(f)(*ptr);
      }
    };
  }

  tsl::AsyncValueRef<T> promise_;

  // Function that is called before a thread starts blocking on the promise.
  FutureHelpers::OnBlockStart on_block_start_;
  // Function that is called after a thread finishes blocking on the promise.
  FutureHelpers::OnBlockEnd on_block_end_;
};

template <typename T>
class PromiseBase {
 public:
  PromiseBase() = default;

  PromiseBase(PromiseBase&& other) = default;
  PromiseBase& operator=(PromiseBase&& other) = default;

  ~PromiseBase() {
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
  //
  // We use this API only when we are exclusive owner of the promise and can
  // guarantee that it didn't escape to other threads via pointers. Otherwise,
  // this is best effort check, because it uses two atomic operations and is
  // not atomic itself.
  bool IsUniqueReference() const {
    CHECK(promise_ && !promise_.GetAsyncValue()->IsIndirect())
        << "Promise must wrap a concrete async value";
    return promise_.GetAsyncValue()->NumRef() == 1 && !promise_.HasWaiter();
  }

 protected:
  friend class ::tsl::Future<internal::future_type_t<T>>;

  explicit PromiseBase(tsl::AsyncValueRef<T> promise)
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

// A type predicate to check if a type combination of `R` and `U` is
// valid for `Future<T>::Map(...)` methods defined below.
template <typename R, typename U>
struct IsMappable : public std::is_constructible<R, U> {};
template <>
struct IsMappable<void, void> : public std::true_type {};
template <>
struct IsMappable<void, absl::Status> : public std::true_type {};
template <typename R, typename U>
struct IsMappable<R, absl::StatusOr<U>> : public std::is_constructible<R, U> {};

// A pre C++20 "concept" that checks if `R` and `U` are mappable types.
template <typename R, typename U>
using Mappable = std::enable_if_t<IsMappable<R, U>::value>;

// Automatic type inference for the result type of `Future<T>::Map(...)` is
// based on the result type of `f` functor:
//
// - `void`              to `Future<>`
// - `absl::Status`      to `Future<>`
// - `absl::StatusOr<T>` to `Future<T>`
// - `R`                 to `Future<R>` (default)
//
// clang-format off
template <typename R> struct MapResult                    { using T = R; };
template <>           struct MapResult<void>              { using T = void; };
template <>           struct MapResult<absl::Status>      { using T = void; };
template <typename R> struct MapResult<absl::StatusOr<R>> { using T = R; };
// clang-format on

template <typename R>
using map_result_t = typename MapResult<R>::T;  // NOLINT

template <typename T>
class PromiseMaker;

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

  static_assert(!internal::is_status_v<T>,
                "Use Future<> specialization for stateless futures");

  static_assert(
      !internal::IsStatusOr<T>::value,
      "Future<T> already has an implicit absl::StatusOr<T> semantics");

 public:
  Future() = default;

  // Constructs an immediately available future with the given value.
  template <
      int&... ExplicitParameterBarrier, typename U,
      std::enable_if_t<std::is_convertible_v<U, absl::StatusOr<T>>>* = nullptr>
  Future(U&& value)  // NOLINT
      : Base(std::forward<U>(value)) {}

  // Constructs and immediately available future from the given value.
  template <
      int&... ExplicitParameterBarrier, typename U,
      std::enable_if_t<std::is_constructible_v<T, U> &&
                       !std::is_convertible_v<U, absl::StatusOr<T>>>* = nullptr>
  explicit Future(U&& value) : Base(std::forward<U>(value)) {}

  using Base::Await;
  using Base::Detach;
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
  // Supported `R` and `U` type combinations:
  //
  // - `Future<>`  from `(const T&) -> void`
  // - `Future<>`  from `(const T&) -> absl::Status`
  // - `Future<R>` from `(const T&) -> absl::StatusOr<U>`
  // - `Future<R>` from `(const T&) -> U`
  //
  // See `Map` functor type inference defined below for more details.
  template <typename R, int&... ExplicitParameterBarrier, typename F,
            typename U = std::invoke_result_t<F, const T&>,
            internal::Mappable<R, U>* = nullptr>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE Future<R> Map(F&& f) const&;

  // A `Map` overload that invokes `f` on the given `executor`.
  template <typename R, int&... ExplicitParameterBarrier, typename F,
            typename U = std::invoke_result_t<F, const T&>,
            internal::Mappable<R, U>* = nullptr>
  [[nodiscard]] Future<R> Map(Executor& executor, F&& f) const&;

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
  // Supported `R` and `U` type combinations: (*)
  //
  // - `Future<>`  from `(T) -> void`
  // - `Future<>`  from `(T) -> absl::Status`
  // - `Future<R>` from `(T) -> absl::StatusOr<U>`
  // - `Future<R>` from `(T) -> U`
  //
  // See `Map` functor type inference defined below for more details.
  //
  // (*) For copyable type `T` functor `f` is called with `const T&` reference.
  template <typename R, int&... ExplicitParameterBarrier, typename F,
            typename U = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>,
            internal::Mappable<R, U>* = nullptr>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE Future<R> Map(F&& f) &&;

  // A `Map` overload that invokes `f` on the given `executor`.
  template <typename R, int&... ExplicitParameterBarrier, typename F,
            typename U = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>,
            internal::Mappable<R, U>* = nullptr>
  [[nodiscard]] Future<R> Map(Executor& executor, F&& f) &&;

  // A `Map` overload that automatically infers the type of result from `f`:
  //
  // - `R` is `absl::Status`      -> Future<>
  // - `R` is `absl::StatusOr<T>` -> Future<T>
  // - `R` is any other type      -> Future<R>
  //
  template <int&... ExplicitParameterBarrier, typename F,
            typename R = std::invoke_result_t<F, const T&>>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE auto Map(F&& f) const& {
    return Map<internal::map_result_t<R>>(std::forward<F>(f));
  }

  // A `Map` overload that invokes `f` on the given `executor`.
  template <int&... ExplicitParameterBarrier, typename F,
            typename R = std::invoke_result_t<F, const T&>>
  [[nodiscard]] auto Map(Executor& executor, F&& f) const& {
    return Map<internal::map_result_t<R>>(executor, std::forward<F>(f));
  }

  // A `Map` overload that automatically infers the type of result from `f`.
  //
  // - `R` is `absl::Status`      -> Future<>
  // - `R` is `absl::StatusOr<T>` -> Future<T>
  // - `R` is any other type      -> Future<R>
  //
  template <int&... ExplicitParameterBarrier, typename F,
            typename R = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE auto Map(F&& f) && {
    return std::move(*this).template Map<internal::map_result_t<R>>(
        std::forward<F>(f));
  }

  // A `Map` overload that invokes `f` on the given `executor`.
  template <int&... ExplicitParameterBarrier, typename F,
            typename R = std::invoke_result_t<
                F, std::conditional_t<is_move_only, T, const T&>>>
  [[nodiscard]] auto Map(Executor& executor, F&& f) && {
    return std::move(*this).template Map<internal::map_result_t<R>>(
        executor, std::forward<F>(f));
  }

  // Flattens a `Future<Future<T>>` to `Future<T>`
  template <typename U = T, std::enable_if_t<internal::IsFuture<U>::value &&
                                             !is_move_only>* = nullptr>
  Future<internal::future_type_t<typename U::value_type>> Flatten() const&;

  // Flattens a `Future<Future<T>>` to `Future<T>`
  template <typename U = T,
            std::enable_if_t<internal::IsFuture<U>::value>* = nullptr>
  Future<internal::future_type_t<typename U::value_type>> Flatten() &&;

 private:
  friend class FutureHelpers;
  friend class ::tsl::Promise<T>;
  friend class internal::PromiseMaker<T>;

  // Wraps a map functor into a callback compatible with Future<>::OnReady.
  template <typename R, typename U, bool rvalue = false, typename F>
  static auto SetPromise(Promise<R> promise, F&& f);

  // Bring FutureBase constructors in scope.
  using Base::Base;

  // Constructor for unavailable future that will be fulfilled later via the
  // promise object.
  //
  // - on_block_start is called before Await starts to block.
  // - on_block_end is called after Await finishes blocking.
  Future(const internal::PromiseBase<absl::StatusOr<T>>& promise,
         FutureHelpers::OnBlockStart on_block_start,
         FutureHelpers::OnBlockEnd on_block_end)
      : Base(promise.ref(), std::move(on_block_start),
             std::move(on_block_end)) {}
};

template <typename T>
class [[nodiscard]] Promise : public internal::PromiseBase<absl::StatusOr<T>> {
  using Base = internal::PromiseBase<absl::StatusOr<T>>;

 public:
  Promise() = default;

  // Sets the value of the promise. Must be called at most once.
  //
  // After Set is called, value will be delivered to waiters on the Future
  // constructed from a promise, via blocking or callbacks.
  void Set(absl::StatusOr<T> value) { Base::emplace(std::move(value)); }

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
  template <int&... ExplicitParameterBarrier, typename U = void,
            std::enable_if_t<std::is_copy_constructible_v<T> &&
                             std::is_void_v<U>>* = nullptr>
  [[nodiscard]] Future<T> future(
      FutureHelpers::OnBlockStart on_block_start = nullptr,
      FutureHelpers::OnBlockEnd on_block_end = nullptr) const {
    return Future<T>(*this, std::move(on_block_start), std::move(on_block_end));
  }

 private:
  friend class Future<T>;
  friend class internal::PromiseMaker<T>;

  explicit Promise(tsl::AsyncValueRef<absl::StatusOr<T>> promise)
      : Base(std::move(promise)) {}
};

// Future<void> specialization for communicating stateless events.
//
// See Future<T> documentation above for more details.
template <>
class Future<void> : public internal::FutureBase<absl::Status> {
  using Base = internal::FutureBase<absl::Status>;

 public:
  Future() = default;

  // Constructor for a future that is immediately ready with a given status.
  // For futures that are immediately ready with OK status, we use a global non
  // reference-counted async value that avoids heap allocation and reference
  // counting operations on a hot path.
  Future(absl::Status status)  // NOLINT
      : Base(ABSL_PREDICT_TRUE(status.ok())
                 ? ready_promise_->AsRef()
                 : tsl::MakeAvailableAsyncValueRef<absl::Status>(
                       std::move(status)),
             /*on_block_start=*/nullptr, /*on_block_end=*/nullptr) {}

  // Support implicit construction from immediate `U` convertible to
  // `absl::Status`.
  template <int&... ExplicitParameterBarrier, typename U,
            std::enable_if_t<std::is_convertible_v<U, absl::Status>>* = nullptr>
  Future(U&& status)  // NOLINT
      : Future(absl::Status(std::forward<U>(status))) {}

  using Base::Await;
  using Base::BlockUntilReady;
  using Base::Detach;
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
  // Supported `R` and `U` type combinations:
  //
  // - `Future<>`  from `() -> void`
  // - `Future<>`  from `() -> absl::Status`
  // - `Future<R>` from `() -> absl::StatusOr<U>`
  // - `Future<R>` from `() -> U`
  //
  // See `Map` functor type inference defined below for more details.
  template <typename R, int&... ExplicitParameterBarrier, typename F,
            typename U = std::invoke_result_t<F>,
            internal::Mappable<R, U>* = nullptr>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE Future<R> Map(F&& f) const;

  // A `Map` overload that invokes `f` on the given `executor`.
  template <typename R, int&... ExplicitParameterBarrier, typename F,
            typename U = std::invoke_result_t<F>,
            internal::Mappable<R, U>* = nullptr>
  [[nodiscard]] Future<R> Map(Executor& executor, F&& f) const;

  // A `Map` overload that automatically infers the type of result from `f`:
  //
  // - `R` is `absl::Status`      -> Future<>
  // - `R` is `absl::StatusOr<T>` -> Future<T>
  // - `R` is any other type      -> Future<R>
  //
  // Functor `f` will be invoked on a thread that sets the promise value,
  // or in the caller thread if the future is already available.
  template <int&... ExplicitParameterBarrier, typename F,
            typename R = std::invoke_result_t<F>>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE auto Map(F&& f) const {
    return Map<internal::map_result_t<R>>(std::forward<F>(f));
  }

  // A `Map` overload that invokes `f` on the given `executor`.
  template <int&... ExplicitParameterBarrier, typename F,
            typename R = std::invoke_result_t<F>>
  [[nodiscard]] auto Map(Executor& executor, F&& f) const {
    return Map<internal::map_result_t<R>>(executor, std::forward<F>(f));
  }

  // Returns an Future<R> that is constructed from the given value. If *this
  // completes with an error, returned future will also be an error.
  //
  // Sample usage: make buffer available when copy is complete
  //
  //   std::unique_ptr<Buffer> buffer = AllocateDestinationBuffer();
  //   Future<> future = CopyToBuffer(buffer, ...);
  //   future.MapTo(std::move(buffer));
  //
  template <typename R>
  [[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE auto MapTo(R&& value) const {
    return Map<absl::remove_cvref_t<R>>(
        [value = std::forward<R>(value)]() mutable {
          return std::move(value);
        });
  }

 private:
  friend class FutureHelpers;
  friend class ::tsl::Promise<void>;
  friend class internal::PromiseMaker<void>;

  // Wraps a map functor into a callback compatible with Future<>::OnReady.
  template <typename R, typename U, typename F>
  static auto SetPromise(Promise<R> promise, F&& f);

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
  Future(const internal::PromiseBase<absl::Status>& promise,
         FutureHelpers::OnBlockStart on_block_start,
         FutureHelpers::OnBlockEnd on_block_end)
      : Base(promise.ref(), std::move(on_block_start),
             std::move(on_block_end)) {}
};

template <>
class [[nodiscard]] Promise<void> : public internal::PromiseBase<absl::Status> {
  using Base = internal::PromiseBase<absl::Status>;

 public:
  Promise() = default;

  // Sets the promise completed with a given status. Must be called at most
  // once.
  //
  // After Set is called, completion event will be delivered to waiters on the
  // Future constructed from a promise, via blocking or callbacks.
  void Set(absl::Status status = absl::OkStatus()) {
    Base::emplace(std::move(status));
  }

  // A helper function to convert move-only Promise to shared_ptr, which is
  // useful when the promise has to be captured by a std::function.
  std::shared_ptr<Promise> ToShared() && {
    return std::make_shared<Promise>(std::move(*this));
  }

  // Returns a future associated with the promise.
  [[nodiscard]] Future<> future(
      FutureHelpers::OnBlockStart on_block_start = nullptr,
      FutureHelpers::OnBlockEnd on_block_end = nullptr) const {
    return Future<>(*this, std::move(on_block_start), std::move(on_block_end));
  }

 private:
  friend class Future<void>;
  friend class internal::PromiseMaker<void>;

  explicit Promise(tsl::AsyncValueRef<absl::Status> promise)
      : Base(std::move(promise)) {}
};

namespace internal {

// Helper class to access private future/promise constructors.
template <typename T>
class PromiseMaker {
 public:
  static std::pair<Promise<T>, Future<T>> Make(
      FutureHelpers::OnBlockStart on_block_start,
      FutureHelpers::OnBlockEnd on_block_end) {
    Promise<T> promise(tsl::MakeUnconstructedAsyncValueRef<
                       typename tsl::Future<T>::value_type>());
    Future<T> future(promise, std::move(on_block_start),
                     std::move(on_block_end));
    return std::make_pair(std::move(promise), std::move(future));
  }
};

}  // namespace internal

// Returns a pair of connected Promise and Future<>. Setting the returned
// promise will fulfill the connected future and will run pending callbacks in
// the caller thread.
//
// - on_block_start is called before Await starts to block.
// - on_block_end is called after Await finishes blocking.
template <typename T = void>
ABSL_ATTRIBUTE_ALWAYS_INLINE std::pair<Promise<T>, Future<T>> MakePromise(
    FutureHelpers::OnBlockStart on_block_start = nullptr,
    FutureHelpers::OnBlockEnd on_block_end = nullptr) {
  return ::tsl::internal::PromiseMaker<T>::Make(std::move(on_block_start),
                                                std::move(on_block_end));
}

// Returns a pair of connected Promise and Future<T>. Setting the returned
// promise will fulfill the connected future and will run all pending
// callbacks on the given `executor`. If the future is ready when `OnReady` or
// `Map` is called, then the callback will be executed immediately in the
// caller thread. Users can explicitly override executor by using `OnReady`
// and `Map` overloads that accept another executor instance.
//
// - on_block_start is called before Await starts to block.
// - on_block_end is called after Await finishes blocking.
template <typename T = void>
ABSL_ATTRIBUTE_ALWAYS_INLINE std::pair<Promise<T>, Future<T>> MakePromise(
    Executor& executor, FutureHelpers::OnBlockStart on_block_start = nullptr,
    FutureHelpers::OnBlockEnd on_block_end = nullptr) {
  auto [promise, future] =
      MakePromise<T>(std::move(on_block_start), std::move(on_block_end));
  return std::make_pair(std::move(promise), std::move(future).Detach(executor));
}

// Returns a future that is constructed from the result of invoking functor
// `f` on the given `executor`.
template <typename T, int&... ExplicitParameterBarrier, typename F,
          typename R = std::invoke_result_t<F>,
          std::enable_if_t<std::is_constructible_v<
              typename tsl::Future<T>::value_type, R>>* = nullptr>
[[nodiscard]] Future<T> MakeFutureOn(Executor& executor, F&& f) {
  auto [promise, future] = MakePromise<T>();
  executor.Execute(
      [promise = std::move(promise), f = std::forward<F>(f)]() mutable {
        promise.Set(std::move(f)());
      });
  return std::move(future);
}

// A `MakeFutureOn` overload that automatically infers the type of the future:
//
// - `T` is `void`              -> Future<>
// - `R` is `absl::Status`      -> Future<>
// - `R` is `absl::StatusOr<T>` -> Future<T>
// - `R` is any other type      -> Future<R>
template <int&... ExplicitParameterBarrier, typename F,
          typename R = std::invoke_result_t<F>>
[[nodiscard]] auto MakeFutureOn(Executor& executor, F&& f) {
  return MakeFutureOn<internal::map_result_t<R>>(executor, std::forward<F>(f));
}

//===----------------------------------------------------------------------===//
// internal::FutureBase<T> implementation.
//===----------------------------------------------------------------------===//

namespace internal {

template <typename T, bool is_move_only>
template <int&... ExplicitParameterBarrier, typename U,
          std::enable_if_t<!is_move_only && std::is_void_v<U>>*>
Future<future_type_t<T>> FutureBase<T, is_move_only>::Detach(
    Executor& executor) const& {
  if (ABSL_PREDICT_FALSE(IsReady())) {
    return Future<future_type_t<T>>(promise_, on_block_start_, on_block_end_);
  }

  RCReference<IndirectAsyncValue> detached = MakeIndirectAsyncValue<T>();
  promise_.AndThen([&executor, detached, ptr = promise_.AsPtr()] {
    // If we hold the last reference to the detached promise, then we can safely
    // forward it to the available value without using an executor, as we know
    // that it will not execute any callbacks in the caller thread.
    if (ABSL_PREDICT_FALSE(detached->NumRef() == 1 && !detached->HasWaiter())) {
      detached->ForwardTo(ptr.CopyRCRef());
    } else {
      executor.Execute(absl::bind_front(&IndirectAsyncValue::ForwardTo,
                                        std::move(detached), ptr.CopyRCRef()));
    }
  });
  return Future<future_type_t<T>>(AsyncValueRef<T>(std::move(detached)),
                                  on_block_start_, on_block_end_);
}

template <typename T, bool is_move_only>
Future<future_type_t<T>> FutureBase<T, is_move_only>::Detach(
    Executor& executor) && {
  if (ABSL_PREDICT_FALSE(IsReady())) {
    return Future<future_type_t<T>>(std::move(promise_),
                                    std::move(on_block_start_),
                                    std::move(on_block_end_));
  }

  AsyncValuePtr<T> ptr = promise_.AsPtr();
  RCReference<IndirectAsyncValue> detached = MakeIndirectAsyncValue<T>();
  ptr.AndThen([&executor, detached, ref = std::move(promise_)]() mutable {
    // If we hold the last reference to the detached promise, then we can safely
    // forward it to the available value without using an executor, as we know
    // that it will not execute any callbacks in the caller thread.
    if (ABSL_PREDICT_FALSE(detached->NumRef() == 1 && !detached->HasWaiter())) {
      detached->ForwardTo(std::move(ref));
    } else {
      executor.Execute(absl::bind_front(&IndirectAsyncValue::ForwardTo,
                                        std::move(detached), std::move(ref)));
    }
  });
  return Future<future_type_t<T>>(AsyncValueRef<T>(std::move(detached)),
                                  std::move(on_block_start_),
                                  std::move(on_block_end_));
}

template <typename T, bool is_move_only>
Future<> FutureBase<T, is_move_only>::GetReadyFuture() const {
  auto [promise, future] = MakePromise<>();
  promise_.AndThen(
      [ptr = promise_.AsPtr(), promise = std::move(promise)]() mutable {
        if constexpr (internal::is_status_v<T>) {
          promise.Set(*ptr);
        } else {
          promise.Set(ptr->status());
        }
      });
  return std::move(future);
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Future<T> implementation.
//===----------------------------------------------------------------------===//

template <typename T>
template <typename R, int&... ExplicitParameterBarrier, typename F, typename U,
          internal::Mappable<R, U>*>
[[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE Future<R> Future<T>::Map(
    F&& f) const& {
  // If `*this` is ready, construct the mapped future immediately.
  if (ABSL_PREDICT_TRUE(Base::promise().IsAvailable())) {
    const absl::StatusOr<T>& value = *Base::promise();

    // Short-circuit and forward existing error to the mapped future.
    if (ABSL_PREDICT_FALSE(!value.ok())) {
      return Future<R>(value.status());
    }

    // Construct the result future available with a result of invoking `f`.
    if constexpr (std::is_void_v<U>) {
      return Future<R>((f(*value), absl::OkStatus()));
    } else {
      return Future<R>(f(*value));
    }
  }

  // If `*this` is not ready yet, we need to create a new promise and fulfill
  // it with a result of `f` when `*this` becomes ready.
  auto [promise, future] = ::tsl::MakePromise<R>();
  OnReady(SetPromise<R, U>(std::move(promise), std::forward<F>(f)));
  return std::move(future);
}

template <typename T>
template <typename R, int&... ExplicitParameterBarrier, typename F, typename U,
          internal::Mappable<R, U>*>
[[nodiscard]] Future<R> Future<T>::Map(Executor& executor, F&& f) const& {
  auto [promise, future] = ::tsl::MakePromise<R>();

  OnReady([&executor, f = std::forward<F>(f), promise = std::move(promise),
           ptr = Base::promise()](const absl::StatusOr<T>& value) mutable {
    // Do not submit a task to the executor if the result is unused.
    if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
      promise.Set(Base::AbortedError());
      return;
    }

    // Extend the lifetime of the underlying async value storage by copying
    // the reference to it, to avoid use-after-free inside the `f` functor.
    executor.Execute([&value, ref = ptr.CopyRef(), f = std::move(f),
                      promise = std::move(promise)]() mutable {
      SetPromise<R, U>(std::move(promise), std::move(f))(value);
    });
  });

  return std::move(future);
}

template <typename T>
template <typename R, int&... ExplicitParameterBarrier, typename F, typename U,
          internal::Mappable<R, U>*>
[[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE Future<R> Future<T>::Map(F&& f) && {
  // If `*this` is ready, construct the mapped future immediately.
  if (ABSL_PREDICT_TRUE(Base::promise().IsAvailable())) {
    // For copyable types bind to const reference, so that we don't
    // accidentally move the value from the underlying async value storage.
    using Value = std::conditional_t<is_move_only, absl::StatusOr<T>&,
                                     const absl::StatusOr<T>&>;
    Value value = *Base::promise();

    // Short-circuit and forward existing error to the mapped future.
    if (ABSL_PREDICT_FALSE(!value.ok())) {
      return Future<R>(value.status());
    }

    // Construct the result future available with a result of invoking `f`.
    if constexpr (std::is_void_v<U>) {
      return Future<R>((f(std::move(*value)), absl::OkStatus()));
    } else {
      return Future<R>(f(std::move(*value)));
    }
  }

  // If `*this` is not ready yet, we need to create a new promise and fulfill
  // it with a result of `f` when `*this` becomes ready.
  auto [promise, future] = ::tsl::MakePromise<R>();
  std::move(*this).OnReady(SetPromise<R, U, /*rvalue=*/true>(
      std::move(promise), std::forward<F>(f)));
  return std::move(future);
}

template <typename T>
template <typename R, int&... ExplicitParameterBarrier, typename F, typename U,
          internal::Mappable<R, U>*>
[[nodiscard]] Future<R> Future<T>::Map(Executor& executor, F&& f) && {
  auto [promise, future] = ::tsl::MakePromise<R>();

  using Value = std::conditional_t<is_move_only, absl::StatusOr<T>,
                                   const absl::StatusOr<T>&>;
  std::move(*this).OnReady([&executor, f = std::forward<F>(f),
                            promise = std::move(promise),
                            ptr = Base::promise()](Value value) mutable {
    // Do not submit a task to the executor if the result is unused.
    if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
      promise.Set(Base::AbortedError());
      return;
    }

    // For move-only types pass by value to the executor callback, and for
    // copyable types pass by const reference to avoid accidental copies. For
    // values passed by reference extend the lifetime of the underlying async
    // value storage by copying the reference to it, to avoid use-after-free
    // inside the `f` functor.
    if constexpr (is_move_only) {
      executor.Execute([value = std::move(value), f = std::move(f),
                        promise = std::move(promise)]() mutable {
        SetPromise<R, U, /*rvalue=*/true>(std::move(promise),
                                          std::move(f))(std::move(value));
      });
    } else {
      executor.Execute([&value, ref = ptr.CopyRef(), f = std::move(f),
                        promise = std::move(promise)]() mutable {
        SetPromise<R, U, /*rvalue=*/true>(std::move(promise),
                                          std::move(f))(value);
      });
    }
  });

  return std::move(future);
}

template <typename T>
template <typename U, std::enable_if_t<internal::IsFuture<U>::value &&
                                       !Future<T>::is_move_only>*>
[[nodiscard]] Future<internal::future_type_t<typename U::value_type>>
Future<T>::Flatten() const& {
  using R = internal::future_type_t<typename U::value_type>;
  auto [promise, future] = MakePromise<R>();

  // For const& API call we always get nested futures and values by reference.
  using NestedFuture = const absl::StatusOr<Future<R>>&;
  using Value = const absl::StatusOr<R>&;

  OnReady([promise = std::move(promise)](NestedFuture nested_future) mutable {
    // Immediately forward error to flatten future.
    if (ABSL_PREDICT_FALSE(!nested_future.ok())) {
      promise.Set(nested_future.status());
      return;
    }

    // Forward nested value when it becomes ready to the promise.
    nested_future->OnReady([promise = std::move(promise)](Value value) mutable {
      promise.Set(value);
    });
  });

  return std::move(future);
}

template <typename T>
template <typename U, std::enable_if_t<internal::IsFuture<U>::value>*>
[[nodiscard]] Future<internal::future_type_t<typename U::value_type>>
Future<T>::Flatten() && {
  using R = internal::future_type_t<typename U::value_type>;
  auto [promise, future] = MakePromise<R>();

  // For move-only futures the nested future and the value moved into the
  // OnReady callback. For copyable futures they are passed by reference,
  // because we don't know how many futures point to the same payload.
  using NestedFuture =
      std::conditional_t<is_move_only, absl::StatusOr<Future<R>>,
                         const absl::StatusOr<Future<R>>&>;
  using Value = std::conditional_t<is_move_only, absl::StatusOr<R>,
                                   const absl::StatusOr<R>&>;

  std::move(*this).OnReady(
      [promise = std::move(promise)](NestedFuture nested_future) mutable {
        // Immediately forward error to flatten future.
        if (ABSL_PREDICT_FALSE(!nested_future.ok())) {
          promise.Set(nested_future.status());
          return;
        }

        // Forward nested value when it becomes ready to the promise.
        std::move(*nested_future)
            .OnReady([promise = std::move(promise)](Value value) mutable {
              promise.Set(std::move(value));
            });
      });

  return std::move(future);
}

template <typename T>
template <typename R, typename U, bool rvalue, typename F>
auto Future<T>::SetPromise(Promise<R> promise, F&& f) {
  // For copyable types bind to const reference, so that we don't
  // accidentally move the value from the underlying async value storage.
  // Move-only types are passed by value into the `OnReady` callback.
  using Value = std::conditional_t<rvalue && is_move_only, absl::StatusOr<T>,
                                   const absl::StatusOr<T>&>;
  return [promise = std::move(promise),
          f = std::forward<F>(f)](Value value) mutable {
    // Do not compute `f` if the result is unused.
    if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
      promise.Set(Base::AbortedError());
      return;
    }

    // Short-circuit and forward existing error to the mapped future.
    if (ABSL_PREDICT_FALSE(!value.ok())) {
      promise.Set(value.status());
      return;
    }

    // Set the result future available with a result of invoking `f`.
    if constexpr (std::is_void_v<U>) {
      promise.Set((f(std::move(*value)), absl::OkStatus()));
    } else {
      promise.Set(f(std::move(*value)));
    }
  };
}

//===----------------------------------------------------------------------===//
// Future<void> implementation.
//===----------------------------------------------------------------------===//

template <typename R, int&... ExplicitParameterBarrier, typename F, typename U,
          internal::Mappable<R, U>*>
[[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE Future<R> Future<void>::Map(
    F&& f) const {
  // If `*this` is ready, construct the mapped future immediately.
  if (ABSL_PREDICT_TRUE(Base::promise().IsAvailable())) {
    // Short-circuit and forward existing error to the mapped future.
    if (ABSL_PREDICT_FALSE(!Base::promise()->ok())) {
      return Future<R>(*Base::promise());
    }

    // Construct the result future available with a result of invoking `f`.
    if constexpr (std::is_void_v<U>) {
      return Future<R>((f(), absl::OkStatus()));
    } else {
      return Future<R>(f());
    }
  }

  // If `*this` is not ready yet, we need to create a new promise and fulfill
  // it with a result of `f` when `*this` becomes ready.
  auto [promise, future] = ::tsl::MakePromise<R>();
  OnReady(SetPromise<R, U>(std::move(promise), std::forward<F>(f)));
  return std::move(future);
}

template <typename R, int&... ExplicitParameterBarrier, typename F, typename U,
          internal::Mappable<R, U>*>
[[nodiscard]] Future<R> Future<void>::Map(Executor& executor, F&& f) const {
  auto [promise, future] = ::tsl::MakePromise<R>();

  OnReady([&executor, f = std::forward<F>(f),
           promise = std::move(promise)](const absl::Status& status) mutable {
    // Do not submit a task to the executor if the result is unused.
    if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
      promise.Set(Base::AbortedError());
      return;
    }

    // Pass `status` by value because it's cheap to copy, instead of extending
    // the lifetime of the underlying async value storage.
    executor.Execute(
        std::bind(SetPromise<R, U>(std::move(promise), std::move(f)), status));
  });

  return std::move(future);
}

template <typename R, typename U, typename F>
auto Future<void>::SetPromise(Promise<R> promise, F&& f) {
  return [promise = std::move(promise),
          f = std::forward<F>(f)](const absl::Status& status) mutable {
    // Do not compute `f` if the result is unused.
    if (ABSL_PREDICT_FALSE(promise.IsUniqueReference())) {
      promise.Set(Base::AbortedError());
      return;
    }

    // Short-circuit and forward existing error to the mapped future.
    if (ABSL_PREDICT_FALSE(!status.ok())) {
      promise.Set(std::move(status));
      return;
    }

    // Set the result future available with a result of invoking `f`.
    if constexpr (std::is_void_v<U>) {
      promise.Set((f(), absl::OkStatus()));
    } else {
      promise.Set(f());
    }
  };
}

//===----------------------------------------------------------------------===//
// JoinFutures implementation.
//===----------------------------------------------------------------------===//

namespace internal {

// A base class for state machines for all `JoinFutures` implementations.
template <typename Derived, typename T = void, typename State = std::monostate>
class JoinFutures {
 public:
  JoinFutures(int32_t size, Promise<T> promise)
      : pending_count_(size), promise_(std::move(promise)) {}

 protected:
  // Updates internal status that tracks futures completed with errors and drops
  // the pending counter. Calls `update_state` callback with a state that
  // keeps intermediate result of completed futures. Calls `complete` callback
  // with a promise that must be completed, the state and the status.
  template <typename UpdateState>
  void Update(absl::Status status, UpdateState&& update_state) {
    if (ABSL_PREDICT_FALSE(!status.ok())) {
      absl::MutexLock lock(mu_);
      if (VLOG_IS_ON(2)) {
        if (!status_.ok() && status.code() != status_.code()) {
          VLOG(2) << "Ignoring status " << status << " because first error was "
                  << status_;
        }
      }
      status_.Update(status);
    }

    // Call the `update_state` callback to give the caller a chance to put
    // completed future value into the `state_` object.
    if constexpr (!std::is_same_v<State, std::monostate>) {
      if (ABSL_PREDICT_TRUE(status.ok())) {
        absl::MutexLock lock(mu_);
        update_state(&state_);
      }
    }

    // Drop the pending futures counter and maybe complete the promise via the
    // user-provided callback.
    int32_t pending_count =
        pending_count_.fetch_sub(1, std::memory_order_acq_rel);
    CHECK_GE(pending_count, 1) << "Pending count can't drop below 0";

    if (pending_count == 1) {
      absl::MutexLock lock(mu_);
      static_cast<Derived&>(*this).Complete(
          std::move(promise_), std::move(status_), std::move(state_));
    }
  };

 private:
  std::atomic<int32_t> pending_count_;
  Promise<T> promise_;

  absl::Mutex mu_;
  absl::Status status_ ABSL_GUARDED_BY(&mu_);
  State state_ ABSL_GUARDED_BY(&mu_);
};

// A state for tracking `JoinFutures` for stateful `Future<T>`.
template <typename T>
class JoinStateful
    : public JoinFutures<JoinStateful<T>, std::vector<T>, std::vector<T>> {
 public:
  using JoinFutures<JoinStateful<T>, std::vector<T>,
                    std::vector<T>>::JoinFutures;

  void OnReady(size_t index, absl::StatusOr<T> value) {
    this->Update(value.status(), [&](std::vector<T>* state) {
      if (state->size() < (1 + index)) {
        state->resize(1 + index);
      }
      state->at(index) = *std::move(value);
    });
  }

  void Complete(Promise<std::vector<T>> promise, absl::Status status,
                std::vector<T> state) {
    if (ABSL_PREDICT_TRUE(status.ok())) {
      promise.Set(std::move(state));
    } else {
      promise.Set(std::move(status));
    }
  }
};

}  // namespace internal

template <typename T, std::enable_if_t<!std::is_void_v<T>>*>
Future<std::vector<T>> JoinFutures(absl::Span<const Future<T>> futures) {
  VLOG(2) << "tsl::JoinFutures: " << futures.size() << " futures";

  if (futures.empty()) {
    return Future<std::vector<T>>({});
  }

  auto [promise, future] = MakePromise<std::vector<T>>();
  auto join = std::make_shared<internal::JoinStateful<T>>(futures.size(),
                                                          std::move(promise));

  for (size_t index = 0; index < futures.size(); ++index) {
    futures[index].OnReady([index, join](absl::StatusOr<T> value) {
      join->OnReady(index, std::move(value));
    });
  }

  return std::move(future);
}

template <typename T, std::enable_if_t<!std::is_void_v<T>>*>
Future<std::vector<T>> JoinFutures(absl::Span<Future<T>> futures) {
  VLOG(2) << "tsl::JoinFutures: " << futures.size() << " futures";

  if (futures.empty()) {
    return Future<std::vector<T>>({});
  }

  auto [promise, future] = MakePromise<std::vector<T>>();
  auto join = std::make_shared<internal::JoinStateful<T>>(futures.size(),
                                                          std::move(promise));

  for (size_t index = 0; index < futures.size(); ++index) {
    std::move(futures[index]).OnReady([index, join](absl::StatusOr<T> value) {
      join->OnReady(index, std::move(value));
    });
  }

  return std::move(future);
}

//===----------------------------------------------------------------------===//
// JoinFutures implementation for statically known arguments.
//===----------------------------------------------------------------------===//

namespace internal {
// A little bit of template meta-programming to figure out the types for
// storing intermediate state during join and the type returned to the caller.
template <typename F>
struct JoinedType;

template <>
struct JoinedType<Future<void>> {
  using state = std::tuple<std::monostate>;
  using result = std::tuple<>;
};

template <typename T>
struct JoinedType<Future<T>> {
  using state = std::tuple<T>;
  using result = std::tuple<T>;
};

template <typename... Futures>
using JoinedTupleState = decltype(std::tuple_cat(
    std::declval<typename JoinedType<Futures>::state>()...));

template <typename... Futures>
using JoinedTupleResult = decltype(std::tuple_cat(
    std::declval<typename JoinedType<Futures>::result>()...));

template <typename T>
auto FilterJoinedTupleMonostate(T&& val) {
  if constexpr (std::is_same_v<std::monostate, std::decay_t<T>>) {
    return std::tuple<>();
  } else {
    return std::make_tuple(std::forward<T>(val));
  }
}

template <typename... Ts>
auto FilterJoinedTuple(std::tuple<Ts...> tuple) {
  return std::apply(
      [](auto&&... args) {
        return std::tuple_cat(
            FilterJoinedTupleMonostate(std::forward<decltype(args)>(args))...);
      },
      std::move(tuple));
}

template <typename R, typename Tuple, typename = void>
struct JoinFromTupleConstructible : std::false_type {};

template <typename R, typename... Args>
struct JoinFromTupleConstructible<R, std::tuple<Args...>>
    : std::is_constructible<R, Args...> {};

// A state for tracking `JoinFutures` for statically known futures types.
template <typename Result, typename State>
class JoinStatic
    : public JoinFutures<JoinStatic<Result, State>, Result, State> {
 public:
  using JoinFutures<JoinStatic<Result, State>, Result, State>::JoinFutures;

  template <std::size_t... Is, typename... Futures>
  static void OnReady(std::shared_ptr<JoinStatic> self,
                      std::index_sequence<Is...>, Futures... futures) {
    (std::forward<Futures>(futures).OnReady([self](auto value) {
      self->OnReady(std::integral_constant<size_t, Is>{}, std::move(value));
    }),
     ...);
  }

  template <size_t index>
  void OnReady(std::integral_constant<size_t, index>, absl::Status status) {
    this->Update(status, [&](State* state) {});
  }

  template <size_t index, typename T>
  void OnReady(std::integral_constant<size_t, index>, absl::StatusOr<T> value) {
    this->Update(value.status(), [&](State* state) {
      std::get<index>(*state) = *std::move(value);
    });
  }

  void Complete(Promise<Result> promise, absl::Status status, State state) {
    if (ABSL_PREDICT_TRUE(status.ok())) {
      promise.Set(
          std::make_from_tuple<Result>(FilterJoinedTuple(std::move(state))));
    } else {
      promise.Set(std::move(status));
    }
  }
};

}  // namespace internal

template <typename R, typename... Futures,
          std::enable_if_t<std::conjunction_v<
              internal::IsFuture<std::decay_t<Futures>>...>>*>
auto JoinFutures(Futures&&... futures) {
  using State = internal::JoinedTupleState<std::decay_t<Futures>...>;
  using Result = internal::JoinedTupleResult<std::decay_t<Futures>...>;

  if constexpr (std::is_same_v<Result, std::tuple<>> && std::is_void_v<R>) {
    // All futures have type `Future<>` and return type is `void`, use a more
    // efficient `JoinFutures`.
    return JoinFutures({futures...});

  } else {
    using PromiseResult = std::conditional_t<std::is_void_v<R>, Result, R>;
    static_assert(
        internal::JoinFromTupleConstructible<PromiseResult, Result>::value,
        "PromiseResult must be constructible from he Result tuple");

    // Create a join state machine for the accumulated futures.
    auto [promise, future] = MakePromise<PromiseResult>();
    auto join = std::make_shared<internal::JoinStatic<PromiseResult, State>>(
        sizeof...(futures), std::move(promise));

    using Is = std::make_index_sequence<sizeof...(Futures)>;
    join->OnReady(std::move(join), Is{}, std::forward<Futures>(futures)...);

    return std::move(future);
  }
}

}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_FUTURE_H_
