/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LOCKABLE_H_
#define XLA_SERVICE_LOCKABLE_H_

#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/logging.h"

namespace xla {

// A template that can be specialized to give a human readable name to lockable
// of type `T`.
template <typename T>
struct LockableName {
  static std::string ToString(const T& value) {
    return absl::StrFormat("lockable %p", &value);
  }
};

// An RAII helper for a value of type `T` that requires exclusive access.
template <typename T, typename LockableName = LockableName<T>>
class Lockable {
 public:
  // RAII type that will release the exclusive lock when it is destroyed.
  class Lock {
   public:
    Lock() = default;

    Lock(Lock&& other) {
      lockable_ = other.lockable_;
      other.lockable_ = nullptr;
    }

    Lock& operator=(Lock&& other) {
      lockable_ = other.lockable_;
      other.lockable_ = nullptr;
      return *this;
    }

    ~Lock() {
      if (lockable_) lockable_->Release();
    }

    T& operator*() const { return lockable_->value_; }
    T* operator->() const { return &lockable_->value_; }
    operator bool() const { return lockable_ != nullptr; }  // NOLINT

    std::string ToString() const {
      return lockable_ ? lockable_->ToString() : "<empty lock>";
    }

   private:
    friend class Lockable;
    explicit Lock(Lockable* lockable) : lockable_(lockable) {}
    Lockable* lockable_ = nullptr;
  };

  Lockable() = default;

  explicit Lockable(T value) : value_(std::move(value)) {
    VLOG(2) << "Constructed " << LockableName::ToString(value_);
  }

  template <typename... Args>
  explicit Lockable(Args&&... args) : value_(std::forward<Args>(args)...) {
    VLOG(2) << "Constructed " << LockableName::ToString(value_);
  }

  Lockable(const Lockable&) = delete;
  Lockable& operator=(const Lockable&) = delete;

  ~Lockable() {
    VLOG(2) << "Destroy " << LockableName::ToString(value_);
    absl::MutexLock lock(&mutex_);
    CHECK_EQ(is_unlocked_, true);  // NOLINT
  }

  Lock Acquire() {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(&is_unlocked_));
    VLOG(2) << "Acquired " << LockableName::ToString(value_);
    is_unlocked_ = false;

    return Lock(this);
  }

  Lock TryAcquire() {
    absl::MutexLock lock(&mutex_);

    // Someone already locked this object, return an empty lock.
    if (is_unlocked_ == false) {
      VLOG(2) << "Failed to acquire " << LockableName::ToString(value_);
      return Lock();
    }

    VLOG(2) << "Acquired " << LockableName::ToString(value_);
    is_unlocked_ = false;
    return Lock(this);
  }

  std::string ToString() const { return LockableName::ToString(value_); }

 protected:
  const T& value() const { return value_; }

 private:
  friend class Lock;

  void Release() {
    absl::MutexLock lock(&mutex_);
    VLOG(2) << "Released " << LockableName::ToString(value_);
    CHECK(!is_unlocked_);  // NOLINT
    is_unlocked_ = true;
  }

  T value_;
  absl::Mutex mutex_;
  bool is_unlocked_ ABSL_GUARDED_BY(mutex_) = true;
};

}  // namespace xla

#endif  // XLA_SERVICE_LOCKABLE_H_
