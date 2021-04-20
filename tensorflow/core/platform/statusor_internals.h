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

#ifndef TENSORFLOW_CORE_PLATFORM_STATUSOR_INTERNALS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUSOR_INTERNALS_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace internal_statusor {

class Helper {
 public:
  // Move type-agnostic error handling to the .cc.
  static void HandleInvalidStatusCtorArg(Status*);
  TF_ATTRIBUTE_NORETURN static void Crash(const Status& status);
};

// Construct an instance of T in `p` through placement new, passing Args... to
// the constructor.
// This abstraction is here mostly for the gcc performance fix.
template <typename T, typename... Args>
void PlacementNew(void* p, Args&&... args) {
#if defined(__GNUC__) && !defined(__clang__)
  // Teach gcc that 'p' cannot be null, fixing code size issues.
  if (p == nullptr) __builtin_unreachable();
#endif
  new (p) T(std::forward<Args>(args)...);
}

// Helper base class to hold the data and all operations.
// We move all this to a base class to allow mixing with the appropriate
// TraitsBase specialization.
template <typename T>
class StatusOrData {
  template <typename U>
  friend class StatusOrData;

 public:
  StatusOrData() = delete;

  StatusOrData(const StatusOrData& other) {
    if (other.ok()) {
      MakeValue(other.data_);
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  StatusOrData(StatusOrData&& other) noexcept {
    if (other.ok()) {
      MakeValue(std::move(other.data_));
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  template <typename U>
  StatusOrData(const StatusOrData<U>& other) {
    if (other.ok()) {
      MakeValue(other.data_);
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  template <typename U>
  StatusOrData(StatusOrData<U>&& other) {
    if (other.ok()) {
      MakeValue(std::move(other.data_));
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  explicit StatusOrData(const T& value) : data_(value) { MakeStatus(); }
  explicit StatusOrData(T&& value) : data_(std::move(value)) { MakeStatus(); }

  explicit StatusOrData(const Status& status) : status_(status) {
    EnsureNotOk();
  }
  explicit StatusOrData(Status&& status) : status_(std::move(status)) {
    EnsureNotOk();
  }

  StatusOrData& operator=(const StatusOrData& other) {
    if (this == &other) return *this;
    if (other.ok())
      Assign(other.data_);
    else
      Assign(other.status_);
    return *this;
  }

  StatusOrData& operator=(StatusOrData&& other) {
    if (this == &other) return *this;
    if (other.ok())
      Assign(std::move(other.data_));
    else
      Assign(std::move(other.status_));
    return *this;
  }

  ~StatusOrData() {
    if (ok()) {
      status_.~Status();
      data_.~T();
    } else {
      status_.~Status();
    }
  }

  void Assign(const T& value) {
    if (ok()) {
      data_.~T();
      MakeValue(value);
    } else {
      MakeValue(value);
      status_ = Status::OK();
    }
  }

  void Assign(T&& value) {
    if (ok()) {
      data_.~T();
      MakeValue(std::move(value));
    } else {
      MakeValue(std::move(value));
      status_ = Status::OK();
    }
  }

  void Assign(const Status& status) {
    Clear();
    status_ = status;
    EnsureNotOk();
  }

  void Assign(Status&& status) {
    Clear();
    // Note that we copy instead of moving the status here so that
    // status.~StatusOrData() can call ok() without invoking UB.
    status_ = status;
    EnsureNotOk();
  }

  bool ok() const { return status_.ok(); }

 protected:
  // status_ will always be active after the constructor.
  // We make it a union to be able to initialize exactly how we need without
  // waste.
  // Eg. in the copy constructor we use the default constructor of Status in
  // the ok() path to avoid an extra Ref call.
  union {
    Status status_;
  };

  // data_ is active iff status_.ok()==true
  struct Dummy {};
  union {
    // When T is const, we need some non-const object we can cast to void* for
    // the placement new. dummy_ is that object.
    Dummy dummy_;
    T data_;
  };

  void Clear() {
    if (ok()) data_.~T();
  }

  void EnsureOk() const {
    if (!ok()) Helper::Crash(status_);
  }

  void EnsureNotOk() {
    if (ok()) Helper::HandleInvalidStatusCtorArg(&status_);
  }

  // Construct the value (ie. data_) through placement new with the passed
  // argument.
  template <typename Arg>
  void MakeValue(Arg&& arg) {
    internal_statusor::PlacementNew<T>(&dummy_, std::forward<Arg>(arg));
  }

  // Construct the status (ie. status_) through placement new with the passed
  // argument.
  template <typename... Args>
  void MakeStatus(Args&&... args) {
    internal_statusor::PlacementNew<Status>(&status_,
                                            std::forward<Args>(args)...);
  }
};

// Helper base class to allow implicitly deleted constructors and assignment
// operations in StatusOr.
// TraitsBase will explicitly delete what it can't support and StatusOr will
// inherit that behavior implicitly.
template <bool Copy, bool Move>
struct TraitsBase {
  TraitsBase() = default;
  TraitsBase(const TraitsBase&) = default;
  TraitsBase(TraitsBase&&) = default;
  TraitsBase& operator=(const TraitsBase&) = default;
  TraitsBase& operator=(TraitsBase&&) = default;
};

template <>
struct TraitsBase<false, true> {
  TraitsBase() = default;
  TraitsBase(const TraitsBase&) = delete;
  TraitsBase(TraitsBase&&) = default;
  TraitsBase& operator=(const TraitsBase&) = delete;
  TraitsBase& operator=(TraitsBase&&) = default;
};

template <>
struct TraitsBase<false, false> {
  TraitsBase() = default;
  TraitsBase(const TraitsBase&) = delete;
  TraitsBase(TraitsBase&&) = delete;
  TraitsBase& operator=(const TraitsBase&) = delete;
  TraitsBase& operator=(TraitsBase&&) = delete;
};

}  // namespace internal_statusor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUSOR_INTERNALS_H_
