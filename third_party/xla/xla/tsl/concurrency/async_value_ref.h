/* Copyright 2022 Google LLC. All Rights Reserved.

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

#ifndef XLA_TSL_CONCURRENCY_ASYNC_VALUE_REF_H_
#define XLA_TSL_CONCURRENCY_ASYNC_VALUE_REF_H_

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/mem.h"

namespace tsl {

// Forward declare owning typed async value pointer.
template <typename T>
class AsyncValueRef;

// Forward declare non-owning typed async value pointer.
template <typename T>
class AsyncValuePtr;

// Constructs a ConcreteAsyncValue in error state with the given status.
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::Status status);

ABSL_DEPRECATED("Use the error async value constructor that takes absl::Status")
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(std::string_view message);

// Constructs an IndirectAsyncValue without forwarding it to anything.
RCReference<IndirectAsyncValue> MakeIndirectAsyncValue();
template <typename T>
RCReference<IndirectAsyncValue> MakeIndirectAsyncValue();

// Forward declare AsyncValueRef constructors.
template <typename T>
AsyncValueRef<T> MakeUnconstructedAsyncValueRef();
template <typename T, typename... Args>
AsyncValueRef<T> MakeConstructedAsyncValueRef(Args&&... args);
template <typename T, typename... Args>
AsyncValueRef<T> MakeAvailableAsyncValueRef(Args&&... args);

// AsyncValueRef<T> is an asynchronous container for a payload of type `T` or an
// error of type `absl::Status`. It is similar to an `absl::StatusOr<T>`, but
// does not require immediate value or error to be constructed. It is a promise
// that at some point in the future it will become concrete and will hold a
// payload of type `T` or an error of type `absl::Status`.
//
//  - Prefer `AsyncValueRef<Chain>` to `AsyncValueRef<absl::Status>`.
//    Instead of a `Chain` it can be any other empty struct to signal that only
//    the potential error is important.
//
//  - Prefer `AsyncValueRef<T>` to `AsyncValueRef<absl::StatusOr<T>>`.
//    Similar to the `absl::StatusOr<T>` async value will be either in error
//    state holding an `absl::Status` error, or in concrete state holding a
//    value of type `T`.
template <typename T>
class AsyncValueRef {
 public:
  // AsyncValueRef<T>::value_type
  using value_type = T;

  AsyncValueRef() = default;

  AsyncValueRef(const AsyncValueRef&) = default;
  AsyncValueRef& operator=(const AsyncValueRef&) = default;

  AsyncValueRef(AsyncValueRef&&) = default;
  AsyncValueRef& operator=(AsyncValueRef&&) = default;

  explicit AsyncValueRef(RCReference<AsyncValue> value)
      : value_(std::move(value)) {}

  template <typename Derived,
            internal::DerivedFrom<Derived, AsyncValue>* = nullptr>
  explicit AsyncValueRef(RCReference<Derived> value)
      : AsyncValueRef(RCReference<AsyncValue>(std::move(value))) {}

  // Support implicit construction from nullptr to empty async value ref.
  AsyncValueRef(std::nullptr_t) {}  // NOLINT

  // Support implicit construction from immediate `Status` error convertible to
  // `absl::Status` (only if payload type is not `absl::Status`, because
  // otherwise it is ambiguous, is it an error or a concrete payload).
  template <typename Status,
            std::enable_if_t<std::is_convertible_v<Status, absl::Status> &&
                             !std::is_same_v<T, absl::Status>>* = nullptr>
  AsyncValueRef(Status&& status)  // NOLINT
      : AsyncValueRef(MakeErrorAsyncValueRef(std::forward<Status>(status))) {}

  // Support implicit conversion from an async value of a derived type.
  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValueRef(AsyncValueRef<Derived> derived)  // NOLINT
      : value_(derived.ReleaseRCRef()) {}

  // Support implicit construction from RCReference<ErrorAsyncValue>.
  AsyncValueRef(RCReference<ErrorAsyncValue> value)  // NOLINT
      : value_(std::move(value)) {}

  AsyncValueRef& operator=(RCReference<ErrorAsyncValue> new_value) {
    value_ = std::move(new_value);
    return *this;
  }

  // Allow implicit conversion to type-erased RCReference<AsyncValue>
  operator RCReference<AsyncValue>() && { return std::move(value_); }  // NOLINT

  // Return true if the AsyncValue is resolved to a concrete value or error.
  bool IsAvailable() const { return value_->IsAvailable(); }
  bool IsUnavailable() const { return value_->IsUnavailable(); }

  // Return true if the AsyncValue contains a concrete value.
  bool IsConcrete() const { return value_->IsConcrete(); }

  // Return true if state is kUnconstructed.
  bool IsUnconstructed() const { return value_->IsUnconstructed(); }

  // Return the stored value. The AsyncValueRef must be available.
  T& get() const { return value_->get<T>(); }

  // Return the stored value as a derived type. The AsyncValueRef must be
  // available.
  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  Derived& get() const {
    return value_->get<Derived>();
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  bool Isa() const {
    // Isa is successful if:
    //   (1) This is no-op cast even if concrete payload has different type.
    //   (2) Type id of a concrete payload matches Derived type id.
    //   (3) Payload is for a special case of ErrorAsyncValue.
    //
    // IMPORTANT: Because AsyncValue can be in unconstructed state we can't rely
    // on `dynamic_cast` (and for similar reason on LLVM casts) and have to
    // rely on type id stored in the async value itself. The downside of this
    // approach that we might return false negatives.
    //
    // Example:
    //
    //   struct A {};
    //   struct B : public A {};
    //   struct C : public C {}
    //
    //   AsyncValueRef<A> ref = MakeUnconstructedAsyncValueRef<C>();
    //
    // In this example `ref.Isa<B>()` will return `false` although `C` can be
    // safely casted to a pointer to its base type `B`, however type id does
    // not have any details about type relationship. This can be fixed by adding
    // extra bits of information to type table and by requiring participating
    // types to register their relationship to base types in terms of their type
    // ids, however there is no such need in practice (so far).
    return value_ && (std::is_same_v<Derived, T> ||                     // (1)
                      value_->IsType<Derived>() ||                      // (2)
                      value_->IsType<DummyValueForErrorAsyncValue>());  // (3)
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValueRef<Derived> Cast() const {
    DCHECK(DynCast<Derived>()) << "Illegal async value cast";
    return AsyncValueRef<Derived>(value_);
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValueRef<Derived> DynCast() const {
    DCHECK(value_) << "Async value must be not null";
    return Isa<Derived>() ? AsyncValueRef<Derived>(value_)
                          : AsyncValueRef<Derived>(nullptr);
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValueRef<Derived> DynCastOrNull() const {
    return value_ ? DynCast<Derived>(value_) : AsyncValueRef<Derived>(nullptr);
  }

  T* operator->() const { return &get(); }

  T& operator*() const { return get(); }

  template <typename Waiter>
  void AndThen(Waiter&& waiter) const {
    AsPtr().AndThen(std::forward<Waiter>(waiter));
  }

  template <typename Waiter>
  void AndThen(AsyncValue::Executor& executor, Waiter&& waiter) const {
    AsPtr().AndThen(executor, std::forward<Waiter>(waiter));
  }

  template <typename R, typename F>
  AsyncValueRef<R> Map(F&& f) {
    return AsPtr().template Map<R>(std::forward<F>(f));
  }

  template <typename R, typename F>
  AsyncValueRef<R> Map(AsyncValue::Executor& executor, F&& f) {
    return AsPtr().template Map<R>(executor, std::forward<F>(f));
  }

  template <typename F>
  auto Map(F&& f) {
    return AsPtr().template Map<F>(std::forward<F>(f));
  }

  template <typename F>
  auto Map(AsyncValue::Executor& executor, F&& f) {
    return AsPtr().template Map<F>(executor, std::forward<F>(f));
  }

  template <typename R, typename F>
  AsyncValueRef<R> TryMap(F&& f) {
    return AsPtr().template TryMap<R>(std::forward<F>(f));
  }

  template <typename R, typename F>
  AsyncValueRef<R> TryMap(AsyncValue::Executor& executor, F&& f) {
    return AsPtr().template TryMap<R>(executor, std::forward<F>(f));
  }

  template <typename F>
  auto TryMap(F&& f) {
    return AsPtr().TryMap(std::forward<F>(f));
  }

  template <typename F>
  auto TryMap(AsyncValue::Executor& executor, F&& f) {
    return AsPtr().TryMap(executor, std::forward<F>(f));
  }

  template <typename F>
  auto FlatMap(F&& f) {
    return AsPtr().FlatMap(std::forward<F>(f));
  }

  template <typename F>
  auto FlatMap(AsyncValue::Executor& executor, F&& f) {
    return AsPtr().FlatMap(executor, std::forward<F>(f));
  }

  // Make the AsyncValueRef available.
  void SetStateConcrete() const { value_->SetStateConcrete(); }

  // Set the stored value. The AsyncValueRef must be unavailable. After this
  // returns, the AsyncValueRef will be available.
  template <typename... Args>
  void emplace(Args&&... args) const {
    value_->emplace<T>(std::forward<Args>(args)...);
  }

  void emplace(absl::StatusOr<T> v) const {
    if (v.ok()) {
      emplace(std::move(*v));
    } else {
      SetError(std::move(v.status()));
    }
  }

  // Return true if this AsyncValueRef represents an error.
  bool IsError() const { return value_->IsError(); }

  // Returns the underlying error. IsError() must be true.
  const absl::Status& GetError() const { return value_->GetError(); }

  // Returns the underlying error, or nullptr if there is none.
  const absl::Status* GetErrorIfPresent() const {
    return value_->GetErrorIfPresent();
  }

  void SetError(absl::Status status) const {
    DCHECK(!status.ok()) << "expected non-ok status";
    return value_->SetError(std::move(status));
  }

  void SetError(std::string_view message) const {
    SetError(absl::InternalError(message));
  }

  explicit operator bool() const { return value_.get() != nullptr; }
  bool operator==(const AsyncValueRef& r) const { return value_ == r.value_; }
  bool operator!=(const AsyncValueRef& r) const { return value_ != r.value_; }

  // Return a raw pointer to the AsyncValue.
  AsyncValue* GetAsyncValue() const { return value_.get(); }

  // Returns a non-owning pointer to the underlying async value.
  AsyncValuePtr<T> AsPtr() const { return AsyncValuePtr<T>(GetAsyncValue()); }

  // Return true if this is the only ref to the AsyncValue.
  // This function requires the internal AsyncValue to be set (value_ !=
  // nullptr).
  bool IsUnique() const { return value_->IsUnique(); }

  // Make an explicit copy of this AsyncValueRef, increasing value_'s refcount
  // by one.
  AsyncValueRef<T> CopyRef() const { return AsyncValueRef(CopyRCRef()); }

  // Make a copy of value_, increasing value_'s refcount by one.
  RCReference<AsyncValue> CopyRCRef() const { return value_; }

  // Release ownership of one reference on the AsyncValue and return a raw
  // pointer to it.
  AsyncValue* release() { return value_.release(); }

  void reset() { value_.reset(); }

  // Transfer ownership of one reference on the AsyncValue to the returned
  // RCReference<AsyncValue>.
  RCReference<AsyncValue> ReleaseRCRef() { return std::move(value_); }

 private:
  RCReference<AsyncValue> value_;
};

// Detects if a type is a specialization of an AsyncValueRef template.
template <typename T>
struct IsAsyncValueRef : std::false_type {};
template <typename T>
struct IsAsyncValueRef<AsyncValueRef<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_async_value_ref_v = IsAsyncValueRef<T>::value;

// Non owning typed pointer for the AsyncValue. Can be cheaply passed around
// when the lifetime of the underlying async value is clear from the context.
// It is the user responsibility to construct an owning AsyncValueRef to extend
// the lifetime of the underlying value if needed.
template <typename T>
class AsyncValuePtr {
  // Detect result types that are `absl::StatusOr<R>` container.
  template <typename R>
  struct IsStatusOr : std::false_type {};
  template <typename R>
  struct IsStatusOr<absl::StatusOr<R>> : std::true_type {};

  // Type predicates for detecting absl::Status-like types.
  template <class R>
  static constexpr bool is_status_v = std::is_same_v<R, absl::Status>;
  template <class R>
  static constexpr bool is_status_or_v = IsStatusOr<R>::value;
  template <class R>
  static constexpr bool is_status_like_v = is_status_v<R> || is_status_or_v<R>;

  // Wait for async value availability: AndThen([] {})
  template <typename Waiter>
  using SimpleWaiter = std::enable_if_t<std::is_invocable_v<Waiter>>;

  // Wait for async value status and value: AndThen([](absl::StatusOr<T*>) {})
  template <typename Waiter>
  using StatusOrWaiter =
      std::enable_if_t<std::is_invocable_v<Waiter, absl::StatusOr<T*>>>;

  // Wait for async value status: AndThen([](absl::Status) {})
  //
  // IMPORTANT: We disable this type of AndThen callback if the payload type is
  // absl::Status because it is ambiguous and confusing: error can be an async
  // value error or a concrete payload of a completed async value. Users should
  // use other types of callbacks to disambiguate the provenance of status.
  template <typename Waiter>
  using StatusWaiter =
      std::enable_if_t<(std::is_invocable_v<Waiter, absl::Status> &&
                        !std::is_invocable_v<Waiter, absl::StatusOr<T*>> &&
                        !is_status_v<T>)>;

  // Because AsyncValue itself is a discriminated union of absl::Status and
  // typed payload (error or value) the use of AsyncValueRef<status-like> is
  // discouraged (work in progress to disable with static assert) and `Map`
  // automatically folds returned status-like object into the returned async
  // value error.

  // Async value map functor: Map<R>([](T& value) -> U);
  //   - R must be constructible from U
  template <typename R, typename U>
  using MapFunctor = std::enable_if_t<std::is_constructible_v<R, U>>;

  // Async value try map functor: TryMap<R>([](T& value) -> absl::StatusOr<U>);
  //   - R must be constructible from U
  template <typename R, typename U>
  using TryMapFunctor =
      std::enable_if_t<!is_status_like_v<R> && is_status_or_v<U> &&
                       std::is_constructible_v<R, typename U::value_type> &&
                       !std::is_constructible_v<R, U>>;

 public:
  // AsyncValuePtr<T>::value_type
  using value_type = T;

  AsyncValuePtr() : value_(nullptr) {}

  explicit AsyncValuePtr(AsyncValue* value) : value_(value) {}
  explicit AsyncValuePtr(const AsyncValueRef<T>& ref)
      : value_(ref.GetAsyncValue()) {}

  AsyncValue* value() const { return value_; }

  AsyncValueRef<T> CopyRef() const { return AsyncValueRef<T>(FormRef(value_)); }

  T& get() const { return value_->template get<T>(); }
  T* operator->() const { return &get(); }
  T& operator*() const { return get(); }

  explicit operator bool() const { return value_ != nullptr; }
  bool operator!=(std::nullptr_t) const { return value_ != nullptr; }
  AsyncValuePtr& operator=(std::nullptr_t) {
    value_ = nullptr;
    return *this;
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  bool Isa() const {
    // Isa is successful if:
    //   (1) This is no-op cast even if concrete payload has different type.
    //   (2) Type id of a concrete payload matches Derived type id.
    //   (3) Payload is for a special case of ErrorAsyncValue.
    return value_ && (std::is_same_v<Derived, T> ||                     // (1)
                      value_->IsType<Derived>() ||                      // (2)
                      value_->IsType<DummyValueForErrorAsyncValue>());  // (3)
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValuePtr<Derived> Cast() const {
    DCHECK(DynCast<Derived>()) << "Illegal async value cast";
    return AsyncValuePtr<Derived>(value_);
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValuePtr<Derived> DynCast() const {
    DCHECK(value_) << "Async value must be not null";
    return Isa<Derived>() ? AsyncValuePtr<Derived>(value_)
                          : AsyncValuePtr<Derived>(nullptr);
  }

  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  AsyncValuePtr<Derived> DynCastOrNull() const {
    return value_ ? DynCast<Derived>(value_) : AsyncValuePtr<Derived>(nullptr);
  }

  bool IsAvailable() const { return value_->IsAvailable(); }
  bool IsUnavailable() const { return value_->IsUnavailable(); }

  bool IsConcrete() const { return value_->IsConcrete(); }
  void SetStateConcrete() const { value_->SetStateConcrete(); }

  template <typename... Args>
  void emplace(Args&&... args) const {
    value_->emplace<T>(std::forward<Args>(args)...);
  }

  bool IsError() const { return value_->IsError(); }

  const absl::Status& GetError() const { return value_->GetError(); }

  void SetError(absl::Status status) const {
    DCHECK(!status.ok()) << "expected non-ok status";
    return value_->SetError(std::move(status));
  }

  // If the AsyncValueRef is available, invokes the `waiter` immediately.
  // Otherwise, invokes the `waiter` when the AsyncValueRef becomes available.
  //
  // Sample usage:
  //
  // async_value_ptr.AndThen([] {
  //   // async_value_ptr is now ready.
  // });
  template <typename Waiter, SimpleWaiter<Waiter>* = nullptr>
  void AndThen(Waiter&& waiter) const {
    value_->AndThen(std::forward<Waiter>(waiter));
  }

  // An overload that executes `waiter` on a user-provided executor.
  template <typename Waiter, SimpleWaiter<Waiter>* = nullptr>
  void AndThen(AsyncValue::Executor& executor, Waiter&& waiter) const {
    value_->AndThen(executor, std::forward<Waiter>(waiter));
  }

  // This AndThen() function takes a functor that takes absl::StatusOr<T*> as
  // argument. This makes it easy for the callback function to use the value of
  // the AsyncValue when it becomes available.
  //
  // Sample usage:
  //
  // async_value_ptr.AndThen([] (absl::StatusOr<T*> status_or) {
  //   // async_value_ptr is now ready and its value/error is in the provided
  //   // `status_or` argument.
  //   if (!status_or.ok()) {
  //      // Handle the error in `status_or.status()`.
  //   } else {
  //      // Handle the value in `*status_or`.
  //   }
  // });
  template <typename Waiter, StatusOrWaiter<Waiter>* = nullptr>
  void AndThen(Waiter&& waiter) const {
    AndThen([waiter = std::forward<Waiter>(waiter), ptr = *this]() mutable {
      if (ABSL_PREDICT_FALSE(ptr.IsError())) {
        return waiter(ptr.GetError());
      } else {
        return waiter(&ptr.get());
      }
    });
  }

  // An overload that executes `waiter` on a user-provided executor.
  template <typename Waiter, StatusOrWaiter<Waiter>* = nullptr>
  void AndThen(AsyncValue::Executor& executor, Waiter&& waiter) const {
    // We don't know when the executor will run the callback, so we need to
    // copy the AsyncValueRef to keep the underlying value alive.
    AndThen(executor,
            [waiter = std::forward<Waiter>(waiter), ref = CopyRef()]() mutable {
              if (ABSL_PREDICT_FALSE(ref.IsError())) {
                return waiter(ref.GetError());
              } else {
                return waiter(&ref.get());
              }
            });
  }

  // This AndThen() function takes a functor that takes an absl::Status as
  // argument. This makes it easy for the callback function to use the error of
  // the AsyncValue when it becomes available. This is useful when the callback
  // function only cares about the error value of the AsyncValue, e.g. for
  // AsyncValueRef<Chain>.
  //
  // Sample usage:
  //
  // async_value_ptr.AndThen([] (absl::Status status) {
  //   // async_value_ptr is now ready and its status is in the provided
  //   // `status` argument.
  //   if (!status.ok()) {
  //     // Handle the error.
  //   } else {
  //     // No error occurred.
  //   }
  // });
  template <typename Waiter, StatusWaiter<Waiter>* = nullptr>
  void AndThen(Waiter&& waiter) const {
    AndThen([waiter = std::forward<Waiter>(waiter), ptr = *this]() mutable {
      if (ABSL_PREDICT_FALSE(ptr.IsError())) {
        return waiter(ptr.GetError());
      } else {
        return waiter(absl::OkStatus());
      }
    });
  }

  // An overload that executes `waiter` on a user-provided executor.
  template <typename Waiter, StatusWaiter<Waiter>* = nullptr>
  void AndThen(AsyncValue::Executor& executor, Waiter&& waiter) const {
    // We don't know when the executor will run the callback, so we need to
    // copy the AsyncValueRef to keep the underlying value alive.
    AndThen(executor,
            [waiter = std::forward<Waiter>(waiter), ref = CopyRef()]() mutable {
              if (ABSL_PREDICT_FALSE(ref.IsError())) {
                return waiter(ref.GetError());
              } else {
                return waiter(absl::OkStatus());
              }
            });
  }

  // Returns and AsyncValueRef<R> that is emplaced from the result of invoking
  // functor `f` with *this value. If *this completes with an error, returned
  // async value will also be an error.
  //
  // Sample usage:
  //
  // async_value_ptr.Map<R>([](T& value) -> U {
  //   return U(value); // R must be constructible from U
  // })
  //
  template <typename R, typename F, typename U = std::invoke_result_t<F, T&>,
            MapFunctor<R, U>* = nullptr>
  AsyncValueRef<R> Map(F&& f) {
    auto result = MakeUnconstructedAsyncValueRef<R>();
    AndThen([f = std::forward<F>(f), result, ptr = *this]() mutable {
      if (ABSL_PREDICT_FALSE(ptr.IsError())) {
        result.SetError(ptr.GetError());
      } else {
        result.emplace(f(*ptr));
      }
    });
    return result;
  }

  // An overload that executes `f` on a user-provided executor.
  template <typename R, typename F, typename U = std::invoke_result_t<F, T&>,
            MapFunctor<R, U>* = nullptr>
  AsyncValueRef<R> Map(AsyncValue::Executor& executor, F&& f) {
    auto result = MakeUnconstructedAsyncValueRef<R>();
    // We don't know when the executor will run the callback, so we need to
    // copy the AsyncValueRef to keep the underlying value alive.
    AndThen(executor,
            [f = std::forward<F>(f), result, ref = CopyRef()]() mutable {
              if (ABSL_PREDICT_FALSE(ref.IsError())) {
                result.SetError(ref.GetError());
              } else {
                result.emplace(f(*ref));
              }
            });
    return result;
  }

  // Returns and AsyncValueRef<R> that is emplaced from the result of invoking
  // functor `f` with *this value. Functor must return an `absl::StatusOr<U>`
  // result that in case of error will be folded into the returned async value
  // as an error. If *this completes with an error, returned async value will
  // also be an error.
  //
  // Sample usage:
  //
  // async_value_ptr.TryMap<R>([](T& value) -> absl::StatusOr<U> {
  //   return absl::StatusOr<U>(U{value}); // R must be constructible from U
  // })
  //
  // If returned status container will have an error status, it will be
  // automatically converted to async value error.
  template <typename R, typename F, typename U = std::invoke_result_t<F, T&>,
            TryMapFunctor<R, U>* = nullptr>
  AsyncValueRef<R> TryMap(F&& f) {
    auto result = MakeUnconstructedAsyncValueRef<R>();
    AndThen([f = std::forward<F>(f), result, ptr = *this]() mutable {
      if (ABSL_PREDICT_FALSE(ptr.IsError())) {
        result.SetError(ptr.GetError());
      } else {
        auto status_or = f(*ptr);
        if (status_or.ok()) {
          result.emplace(std::move(status_or.value()));
        } else {
          result.SetError(status_or.status());
        }
      }
    });
    return result;
  }

  // An overload that executes `f` on a user-provided executor.
  template <typename R, typename F, typename U = std::invoke_result_t<F, T&>,
            TryMapFunctor<R, U>* = nullptr>
  AsyncValueRef<R> TryMap(AsyncValue::Executor& executor, F&& f) {
    auto result = MakeUnconstructedAsyncValueRef<R>();
    // We don't know when the executor will run the callback, so we need to
    // copy the AsyncValueRef to keep the underlying value alive.
    AndThen(executor,
            [f = std::forward<F>(f), result, ref = CopyRef()]() mutable {
              if (ABSL_PREDICT_FALSE(ref.IsError())) {
                result.SetError(ref.GetError());
              } else {
                auto status_or = f(*ref);
                if (status_or.ok()) {
                  result.emplace(std::move(status_or.value()));
                } else {
                  result.SetError(status_or.status());
                }
              }
            });
    return result;
  }

  // A `Map` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<F, T&>>
  auto Map(F&& f) {
    return Map<R>(std::forward<F>(f));
  }

  // A `Map` overload that automatically infers the type of result from `f` and
  // executes `f` on user-provided executor.
  template <typename F, typename R = std::invoke_result_t<F, T&>>
  auto Map(AsyncValue::Executor& executor, F&& f) {
    return Map<R>(executor, std::forward<F>(f));
  }

  // A `TryMap` overload that automatically infers the type of result from `f`.
  template <typename F, typename R = std::invoke_result_t<F, T&>,
            std::enable_if_t<is_status_or_v<R>>* = nullptr>
  auto TryMap(F&& f) {
    return TryMap<typename R::value_type>(std::forward<F>(f));
  }

  // A `TryMap` overload that automatically infers the type of result from `f`
  // and executes `f` on user-provided executor.
  template <typename F, typename R = std::invoke_result_t<F, T&>,
            std::enable_if_t<is_status_or_v<R>>* = nullptr>
  auto TryMap(AsyncValue::Executor& executor, F&& f) {
    return TryMap<typename R::value_type>(executor, std::forward<F>(f));
  }

  // Returns and AsyncValueRef<R> that will be forwarded to the AsyncValueRef
  // returned from a functor.
  //
  // Sample usage:
  //
  // async_value_ptr.FlatMap([](T& value) -> AsyncValueRef<R> {
  //   return LaunchAsyncTask(value);
  // })
  //
  template <typename F, typename R = std::invoke_result_t<F, T&>,
            std::enable_if_t<is_async_value_ref_v<R>>* = nullptr>
  AsyncValueRef<typename R::value_type> FlatMap(F&& f) {
    // If async value is in concrete state, we can immediately call the functor.
    // We don't handle errors here and prefer a generic code path below because
    // error handling is never on a performance critical path.
    if (ABSL_PREDICT_TRUE(IsConcrete())) {
      return f(get());
    }

    auto promise = MakePromise<R>();
    AndThen([f = std::forward<F>(f), promise, ptr = *this]() mutable {
      if (ABSL_PREDICT_FALSE(ptr.IsError())) {
        promise->SetError(ptr.GetError());
      } else {
        promise->ForwardTo(f(*ptr));
      }
    });
    return AsyncValueRef<typename R::value_type>(promise);
  }

  // An overload that executes `f` on a user-provided executor.
  template <typename F, typename R = std::invoke_result_t<F, T&>,
            std::enable_if_t<is_async_value_ref_v<R>>* = nullptr>
  AsyncValueRef<typename R::value_type> FlatMap(AsyncValue::Executor& executor,
                                                F&& f) {
    // We don't have a special handling for concrete values here because
    // we must execute user functor on a separate executor and can't call it in
    // the caller thread.
    auto promise = MakePromise<R>();
    // We don't know when the executor will run the callback, so we need to
    // copy the AsyncValueRef to keep the underlying value alive.
    AndThen(executor,
            [f = std::forward<F>(f), promise, ref = CopyRef()]() mutable {
              if (ABSL_PREDICT_FALSE(ref.IsError())) {
                promise->SetError(ref.GetError());
              } else {
                promise->ForwardTo(f(*ref));
              }
            });
    return AsyncValueRef<typename R::value_type>(promise);
  }

 private:
  // We set a concrete type for indirect async value promise only if the type is
  // final, because otherwise we can forward it later to one of the derived
  // types and this will be a run time error.
  template <typename R>
  RCReference<IndirectAsyncValue> MakePromise() {
    if constexpr (std::is_final_v<typename R::value_type>) {
      return MakeIndirectAsyncValue<typename R::value_type>();
    } else {
      return MakeIndirectAsyncValue();
    };
  }

  AsyncValue* value_;  // doesn't own the async value
};

//===----------------------------------------------------------------------===//
// Functions for awaiting on the async values.
//===----------------------------------------------------------------------===//

template <typename T>
void BlockUntilReady(const AsyncValueRef<T>& ref) {
  BlockUntilReady(ref.GetAsyncValue());
}

template <typename T>
void BlockUntilReady(const AsyncValuePtr<T>& ptr) {
  BlockUntilReady(ptr.value());
}

template <typename T>
void RunWhenReady(absl::Span<const AsyncValueRef<T>> refs,
                  absl::AnyInvocable<void()> callee) {
  absl::InlinedVector<AsyncValue*, 8> values(refs.size());
  for (size_t i = 0; i < refs.size(); ++i) {
    values[i] = refs[i].GetAsyncValue();
  }
  RunWhenReady(values, std::move(callee));
}

template <typename T>
void RunWhenReady(absl::Span<const AsyncValuePtr<T>> ptrs,
                  absl::AnyInvocable<void()> callee) {
  absl::InlinedVector<AsyncValue*, 8> values(ptrs.size());
  for (size_t i = 0; i < ptrs.size(); ++i) {
    values[i] = ptrs[i].value();
  }
  RunWhenReady(values, std::move(callee));
}

//===----------------------------------------------------------------------===//
// LLVM-style type casting library for async value refs and ptrs.
//===----------------------------------------------------------------------===//

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
bool Isa(const AsyncValueRef<T>& ref) {
  return ref.template Isa<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
AsyncValueRef<Derived> Cast(const AsyncValueRef<T>& ref) {
  return ref.template Cast<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
AsyncValueRef<Derived> DynCast(const AsyncValueRef<T>& ref) {
  return ref.template DynCast<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
AsyncValueRef<Derived> DynCastOrNull(const AsyncValueRef<T>& ref) {
  return ref.template DynCastOrNull<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
bool Isa(AsyncValuePtr<T> ptr) {
  return ptr.template Isa<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
AsyncValuePtr<Derived> Cast(AsyncValuePtr<T> ptr) {
  return ptr.template Cast<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
AsyncValuePtr<Derived> DynCast(AsyncValuePtr<T> ptr) {
  return ptr.template DynCast<Derived>();
}

template <typename Derived, typename T,
          internal::DerivedFrom<Derived, T>* = nullptr>
AsyncValuePtr<Derived> DynCastOrNull(AsyncValuePtr<T> ptr) {
  return ptr.template DynCastOrNull<Derived>();
}

//===----------------------------------------------------------------------===//
// Constructing reference-counted async values on the heap.
//===----------------------------------------------------------------------===//

namespace internal {

template <typename T, typename... Args>
T* PlacementConstruct(void* buf, Args&&... args) {
  return new (buf) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
T* AllocateAndConstruct(Args&&... args) {
  void* buf = port::AlignedMalloc(sizeof(T), alignof(T));
  return PlacementConstruct<T, Args...>(buf, std::forward<Args>(args)...);
}

}  // namespace internal

// Construct an empty IndirectAsyncValue with a known type.
template <typename T>
RCReference<IndirectAsyncValue> MakeIndirectAsyncValue() {
  return TakeRef(internal::AllocateAndConstruct<TypedIndirectAsyncValue<T>>());
}

// Allocate an unconstructed AsyncValueRef. The AsyncValueRef should be made
// available later by invoking AsyncValueRef::emplace or
// AsyncValueRef::SetError.
template <typename T>
AsyncValueRef<T> MakeUnconstructedAsyncValueRef() {
  return AsyncValueRef<T>(tsl::TakeRef(
      internal::AllocateAndConstruct<internal::ConcreteAsyncValue<T>>(
          typename internal::ConcreteAsyncValue<T>::UnconstructedPayload{})));
}

// Allocate and construct an AsyncValueRef without making it available for
// consumption. The AsyncValueRef should be made available later by invoking
// AsyncValueRef::SetStateConcrete or AsyncValueRef::SetError.
template <typename T, typename... Args>
AsyncValueRef<T> MakeConstructedAsyncValueRef(Args&&... args) {
  return AsyncValueRef<T>(tsl::TakeRef(
      internal::AllocateAndConstruct<internal::ConcreteAsyncValue<T>>(
          typename internal::ConcreteAsyncValue<T>::ConstructedPayload{},
          std::forward<Args>(args)...)));
}

// Allocate and construct an available AsyncValueRef.
template <typename T, typename... Args>
AsyncValueRef<T> MakeAvailableAsyncValueRef(Args&&... args) {
  return AsyncValueRef<T>(tsl::TakeRef(
      internal::AllocateAndConstruct<internal::ConcreteAsyncValue<T>>(
          typename internal::ConcreteAsyncValue<T>::ConcretePayload{},
          std::forward<Args>(args)...)));
}

//===----------------------------------------------------------------------===//
// Constructing non-reference-counted values in user provided storage.
//===----------------------------------------------------------------------===//

namespace internal {

// Properly sized and aligned storage for allocating async values of given type.
template <typename T>
struct AsyncValueStorage {
  using Payload = ConcreteAsyncValue<T>;

  AsyncValueStorage() = default;

  AsyncValueStorage(const AsyncValueStorage&) = delete;
  AsyncValueStorage& operator=(const AsyncValueStorage&) = delete;

  void* buf() { return &storage[0]; }

  alignas(Payload) std::byte storage[sizeof(Payload)];
};

}  // namespace internal

// Exclusive owner of the non reference-counted async value (e.g. allocated in
// the user provided storage) that is responsible for destructing it. If you'd
// look at `AsyncValueRef` as `std::shared_ptr`, then this is `std::unique_ptr`.
template <typename T>
class AsyncValueOwningRef {
 public:
  AsyncValueOwningRef() = default;
  ~AsyncValueOwningRef() { Destroy(); }

  AsyncValueOwningRef(const AsyncValueOwningRef&) = delete;
  AsyncValueOwningRef& operator=(const AsyncValueOwningRef&) = delete;

  AsyncValueOwningRef& operator=(AsyncValueOwningRef&& other) {
    Destroy();
    std::swap(value_, other.value_);
    return *this;
  }

  AsyncValueOwningRef(AsyncValueOwningRef&& other) {
    Destroy();
    std::swap(value_, other.value_);
  }

  AsyncValueRef<T> AsRef() const { return AsyncValueRef<T>(FormRef(value_)); }
  AsyncValuePtr<T> AsPtr() const { return AsyncValuePtr<T>(value_); }

  T* operator->() const { return &value_->get(); }
  T& operator*() const { return value_->get(); }

 private:
  template <typename U, typename... Args>
  friend AsyncValueOwningRef<U> MakeConstructedAsyncValueRef(
      internal::AsyncValueStorage<U>&, Args&&...);

  template <typename U, typename... Args>
  friend AsyncValueOwningRef<U> MakeAvailableAsyncValueRef(
      internal::AsyncValueStorage<U>&, Args&&...);

  explicit AsyncValueOwningRef(internal::ConcreteAsyncValue<T>* value)
      : value_(value) {}

  void Destroy() {
    if (value_) {
      CallDestructor(value_);
      value_ = nullptr;
    }
  }

  // Work around NVCC compilation error.
  template <typename U>
  void CallDestructor(U* ptr) {
    ptr->~U();
  }

  internal::ConcreteAsyncValue<T>* value_ = nullptr;
};

// Constructs an AsyncValueRef in the provided storage without making it
// available for consumption. The AsyncValueRef should be made available later
// by invoking AsyncValueRef::SetStateConcrete or AsyncValueRef::SetError.
template <typename T, typename... Args>
AsyncValueOwningRef<T> MakeConstructedAsyncValueRef(
    internal::AsyncValueStorage<T>& storage, Args&&... args) {
  return AsyncValueOwningRef<T>(
      internal::PlacementConstruct<internal::ConcreteAsyncValue<T>>(
          storage.buf(),
          typename internal::ConcreteAsyncValue<T>::ConstructedPayload{false},
          std::forward<Args>(args)...));
}

// Construct an available AsyncValueRef in the provided storage.
template <typename T, typename... Args>
AsyncValueOwningRef<T> MakeAvailableAsyncValueRef(
    internal::AsyncValueStorage<T>& storage, Args&&... args) {
  return AsyncValueOwningRef<T>(
      internal::PlacementConstruct<internal::ConcreteAsyncValue<T>>(
          storage.buf(),
          typename internal::ConcreteAsyncValue<T>::ConcretePayload{false},
          std::forward<Args>(args)...));
}

}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_ASYNC_VALUE_REF_H_
