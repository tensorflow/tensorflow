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

#ifndef TENSORFLOW_TSL_CONCURRENCY_ASYNC_VALUE_REF_H_
#define TENSORFLOW_TSL_CONCURRENCY_ASYNC_VALUE_REF_H_

#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/tsl/concurrency/async_value.h"
#include "tensorflow/tsl/concurrency/ref_count.h"

namespace tsl {

// Forward declare non-owning typed async value pointer.
template <typename T>
class AsyncValuePtr;

// RCReference<AsyncValue> wrapper.
//
// AsyncValueRef<T> is an alias for RCReference<AsyncValue> that carries payload
// type information. The user does not need to pass the payload data type to
// get() or emplace().
//
// Like RCReference<AsyncValue>, it represents one reference on the underlying
// AsyncValue. When a callee returns an AsyncValueRef to a caller, the callee
// also transfers their ownership of a reference on the underlying AsyncValue.
template <typename T>
class AsyncValueRef {
 public:
  AsyncValueRef() = default;
  AsyncValueRef(std::nullptr_t) {}  // NOLINT

  explicit AsyncValueRef(RCReference<AsyncValue> value)
      : value_(std::move(value)) {}

  // Support implicit conversion from AsyncValueRef<Derived> to
  // AsyncValueRef<Base>.
  template <typename DerivedT,
            std::enable_if_t<std::is_base_of<T, DerivedT>::value>* = nullptr>
  AsyncValueRef(AsyncValueRef<DerivedT>&& u)  // NOLINT
      : value_(u.ReleaseRCRef()) {}

  // Support implicit conversion from RCReference<AsyncValue>.
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

  // Return the stored value as a subclass type. The AsyncValueRef must be
  // available.
  template <typename SubclassT,
            std::enable_if_t<std::is_base_of<T, SubclassT>::value>* = nullptr>
  SubclassT& get() const {
    return value_->get<SubclassT>();
  }

  T* operator->() const { return &get(); }

  T& operator*() const { return get(); }

  template <typename WaiterT>
  void AndThen(WaiterT&& waiter) const {
    AsPtr().AndThen(std::forward<WaiterT>(waiter));
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
    assert(!status.ok() && "expected non-ok status");
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

// Non owning typed pointer for the AsyncValue. Can be cheaply passed around
// when the lifetime of the underlying async value is clear from the context.
// It is the user responsibility to construct an owning AsyncValueRef to extend
// the lifetime of the underlying value if needed.
template <typename T>
class AsyncValuePtr {
 public:
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
    assert(!status.ok() && "expected non-ok status");
    return value_->SetError(std::move(status));
  }

  // If the AsyncValueRef is available, run the waiter immediately. Otherwise,
  // run the waiter when the AsyncValueRef becomes available.
  //
  // Sample usage:
  //
  // async_value_ref.AndThen([] {
  //   // async_value_ref is now ready.
  // });
  template <typename WaiterT,
            std::enable_if_t<std::is_invocable_v<WaiterT>>* = nullptr>
  void AndThen(WaiterT&& waiter) const {
    value_->AndThen(std::forward<WaiterT>(waiter));
  }

  // This AndThen() function takes a functor that takes absl::StatusOr<T*> as
  // argument. This makes it easy for the callback function to use the value of
  // the AsyncValue when it becomes available.
  //
  // Sample usage:
  //
  // async_value_ref.AndThen([] (absl::StatusOr<T*> status_or) {
  //   // async_value_ref is now ready and its value/error is in the provided
  //   // `status_or` argument.
  //   if (!status_or.ok()) {
  //      // Handle the error in `status_or.status()`.
  //   } else {
  //      // Handle the value in `*status_or`.
  //   }
  // });
  template <typename WaiterT, std::enable_if_t<std::is_invocable_v<
                                  WaiterT, absl::StatusOr<T*>>>* = nullptr>
  void AndThen(WaiterT&& waiter) const {
    AndThen([waiter = std::forward<WaiterT>(waiter), av_ptr = *this]() mutable {
      if (av_ptr.IsError()) {
        return std::forward<WaiterT>(waiter)(av_ptr.GetError());
      } else {
        return std::forward<WaiterT>(waiter)(&av_ptr.get());
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
  // async_value_ref.AndThen([] (absl::Status status) {
  //   // async_value_ref is now ready and its status is in the provided
  //   // `status` argument.
  //   if (!status.ok()) {
  //     // Handle the error.
  //   } else {
  //     // No error occurred.
  //   }
  // });
  template <typename WaiterT,
            std::enable_if_t<
                (std::is_invocable_v<WaiterT, absl::Status> &&
                 !std::is_invocable_v<WaiterT, absl::StatusOr<T*>>)>* = nullptr>
  void AndThen(WaiterT&& waiter) const {
    AndThen([waiter = std::forward<WaiterT>(waiter), av_ptr = *this]() mutable {
      if (av_ptr.IsError()) {
        return std::forward<WaiterT>(waiter)(av_ptr.GetError());
      } else {
        return std::forward<WaiterT>(waiter)(absl::OkStatus());
      }
    });
  }

 private:
  AsyncValue* value_;  // doesn't own the async value
};

// Create a ConcreteAsyncValue in error state with the given status.
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::Status status);

ABSL_DEPRECATED("Use the error async value constructor that takes absl::Status")
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(std::string_view message);

// Construct an empty IndirectAsyncValue, not forwarding to anything.
RCReference<IndirectAsyncValue> MakeIndirectAsyncValue();

//===----------------------------------------------------------------------===//

namespace internal {

template <typename T, typename... Args>
T* PlacementConstruct(void* buf, Args&&... args) {
  return new (buf) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
T* AllocateAndConstruct(Args&&... args) {
  // TODO(ezhulenev): `port::AlignedMalloc` has a different order of arguments!
  void* buf = internal::AlignedAlloc(alignof(T), sizeof(T));
  return PlacementConstruct<T, Args...>(buf, std::forward<Args>(args)...);
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Constructing reference-counted async values on the heap.
//===----------------------------------------------------------------------===//

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

#endif  // TENSORFLOW_TSL_CONCURRENCY_ASYNC_VALUE_REF_H_
