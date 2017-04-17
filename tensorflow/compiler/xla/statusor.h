/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// StatusOr<T> is the union of a Status object and a T
// object. StatusOr models the concept of an object that is either a
// usable value, or an error Status explaining why such a value is
// not present. To this end, StatusOr<T> does not allow its Status
// value to be Status::OK. Furthermore, the value of a StatusOr<T*>
// must not be null. This is enforced by a debug check in most cases,
// but even when it is not, clients must not set the value to null.
//
// The primary use-case for StatusOr<T> is as the return value of a
// function which may fail.
//
// Example client usage for a StatusOr<T>, where T is not a pointer:
//
//  StatusOr<float> result = DoBigCalculationThatCouldFail();
//  if (result.ok()) {
//    float answer = result.ValueOrDie();
//    printf("Big calculation yielded: %f", answer);
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<T*>:
//
//  StatusOr<Foo*> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo(result.ValueOrDie());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<std::unique_ptr<T>>:
//
//  StatusOr<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo = std::move(result.ValueOrDie());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example factory implementation returning StatusOr<T*>:
//
//  StatusOr<Foo*> FooFactory::MakeNewFoo(int arg) {
//    if (arg <= 0) {
//      return tensorflow::InvalidArgument("Arg must be positive");
//    } else {
//      return new Foo(arg);
//    }
//  }
//
// Note that the assignment operators require that destroying the currently
// stored value cannot invalidate the argument; in other words, the argument
// cannot be an alias for the current value, or anything owned by the current
// value.
#ifndef TENSORFLOW_COMPILER_XLA_STATUSOR_H_
#define TENSORFLOW_COMPILER_XLA_STATUSOR_H_

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
template <typename T, bool CopyConstructible>
class TF_MUST_USE_RESULT StatusOr;
#endif

template <typename T,
          bool CopyConstructible = std::is_copy_constructible<T>::value>
class StatusOr {
  template <typename U, bool UC>
  friend class StatusOr;

 public:
  typedef T element_type;

  // Construct a new StatusOr with Status::UNKNOWN status
  StatusOr();

  // Construct a new StatusOr with the given non-ok status. After calling
  // this constructor, calls to ValueOrDie() will CHECK-fail.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return
  // value, so it is convenient and sensible to be able to do 'return
  // Status()' when the return type is StatusOr<T>.
  //
  // REQUIRES: status != Status::OK. This requirement is DCHECKed.
  // In optimized builds, passing Status::OK here will have the effect
  // of passing tensorflow::error::INTERNAL as a fallback.
  StatusOr(Status status);              // NOLINT

  // Construct a new StatusOr with the given value. If T is a plain pointer,
  // value must not be NULL. After calling this constructor, calls to
  // ValueOrDie() will succeed, and calls to status() will return OK.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when the return type is StatusOr<T>.
  //
  // REQUIRES: if T is a plain pointer, value != NULL. This requirement is
  // DCHECKed. In optimized builds, passing a NULL pointer here will have
  // the effect of passing tensorflow::error::INTERNAL as a fallback.
  StatusOr(const T& value);  // NOLINT

  // Copy constructor.
  StatusOr(const StatusOr& other) = default;

  // Conversion copy constructor, T must be copy constructible from U
  template <typename U>
  StatusOr(const StatusOr<U>& other);

  // Assignment operator.
  StatusOr& operator=(const StatusOr& other) = default;

  // Conversion assignment operator, T must be assignable from U
  template <typename U>
  StatusOr& operator=(const StatusOr<U>& other);

  // Move constructor and move-assignment operator.
  StatusOr(StatusOr&& other) = default;
  StatusOr& operator=(StatusOr&& other) = default;

  // Rvalue-reference overloads of the other constructors and assignment
  // operators, to support move-only types and avoid unnecessary copying.
  //
  // Implementation note: we could avoid all these rvalue-reference overloads
  // if the existing lvalue-reference overloads took their arguments by value
  // instead. I think this would also let us omit the conversion assignment
  // operator altogether, since we'd get the same functionality for free
  // from the implicit conversion constructor and ordinary assignment.
  // However, this could result in extra copy operations unless we use
  // std::move to avoid them, and we can't use std::move because this code
  // needs to be portable to C++03.
  StatusOr(T&& value);  // NOLINT
  template <typename U>
  StatusOr(StatusOr<U>&& other);

  // Returns a reference to our status. If this contains a T, then
  // returns Status::OK.
  const Status& status() const { return status_; }

  // Returns this->status().ok()
  bool ok() const { return status_.ok(); }

  // Returns a reference to our current value, or CHECK-fails if !this->ok().
  const T& ValueOrDie() const;
  T& ValueOrDie();

  // Moves our current value out of this object and returns it, or CHECK-fails
  // if !this->ok().
  // Use of this method is discouraged; prefer std::move(statusor.ValueOrDie())
  // instead.
  T ConsumeValueOrDie() { return std::move(ValueOrDie()); }

 private:
  Status status_;
  T value_;
};

// Partial specialization for when T is not copy-constructible. This uses all
// methods from the core implementation, but removes copy assignment and copy
// construction.
template <typename T>
class StatusOr<T, false> : public StatusOr<T, true> {
 public:
  // Remove copies.
  StatusOr(const StatusOr& other) = delete;
  StatusOr& operator=(const StatusOr& other) = delete;
  template <typename U>
  StatusOr(const StatusOr<U>& other) = delete;
  StatusOr(const T& value) = delete;

  // Use the superclass version for other constructors and operators.
  StatusOr() = default;
  StatusOr(StatusOr&& other) = default;
  StatusOr& operator=(StatusOr&& other) = default;
  StatusOr(T&& value)  // NOLINT
      : StatusOr<T, true>::StatusOr(std::move(value)) {}
  StatusOr(Status status)  // NOLINT
      : StatusOr<T, true>::StatusOr(std::move(status)) {}
  template <typename U>
  StatusOr(StatusOr<U>&& other)  // NOLINT
      : StatusOr<T, true>::StatusOr(std::move(other)) {}
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details for StatusOr<T>

namespace internal {

class StatusOrHelper {
 public:
  // Move type-agnostic error handling to the .cc.
  static Status HandleInvalidStatusCtorArg();
  static Status HandleNullObjectCtorArg();
  static void Crash(const Status& status);

  // Customized behavior for StatusOr<T> vs. StatusOr<T*>
  template <typename T>
  struct Specialize;
};

template <typename T>
struct StatusOrHelper::Specialize {
  // For non-pointer T, a reference can never be NULL.
  static inline bool IsValueNull(const T& t) { return false; }
};

template <typename T>
struct StatusOrHelper::Specialize<T*> {
  static inline bool IsValueNull(const T* t) { return t == NULL; }
};

}  // namespace internal

template <typename T, bool CopyConstructible>
inline StatusOr<T, CopyConstructible>::StatusOr()
    : status_(tensorflow::error::UNKNOWN, "") {}

template <typename T, bool CopyConstructible>
inline StatusOr<T, CopyConstructible>::StatusOr(Status status)
    : status_(std::move(status)) {
  if (status_.ok()) {
    status_ = internal::StatusOrHelper::HandleInvalidStatusCtorArg();
  }
}

template <typename T, bool CopyConstructible>
inline StatusOr<T, CopyConstructible>::StatusOr(const T& value)
    : value_(value) {
  if (internal::StatusOrHelper::Specialize<T>::IsValueNull(value)) {
    status_ = internal::StatusOrHelper::HandleNullObjectCtorArg();
  }
}

template <typename T, bool CopyConstructible>
template <typename U>
inline StatusOr<T, CopyConstructible>::StatusOr(const StatusOr<U>& other)
    : status_(other.status_), value_(other.value_) {}

template <typename T, bool CopyConstructible>
inline StatusOr<T, CopyConstructible>::StatusOr(T&& value)
    : value_(std::move(value)) {
  if (internal::StatusOrHelper::Specialize<T>::IsValueNull(value_)) {
    status_ = internal::StatusOrHelper::HandleNullObjectCtorArg();
  }
}

template <typename T, bool CopyConstructible>
template <typename U>
inline StatusOr<T, CopyConstructible>::StatusOr(StatusOr<U>&& other)
    : status_(std::move(other.status_)), value_(std::move(other.value_)) {}

template <typename T, bool CopyConstructible>
inline const T& StatusOr<T, CopyConstructible>::ValueOrDie() const {
  if (!ok()) {
    internal::StatusOrHelper::Crash(status());
  }
  return value_;
}

template <typename T, bool CopyConstructible>
inline T& StatusOr<T, CopyConstructible>::ValueOrDie() {
  if (!status_.ok()) {
    internal::StatusOrHelper::Crash(status());
  }
  return value_;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_STATUSOR_H_
