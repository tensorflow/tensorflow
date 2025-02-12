/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_MUTEX_H_
#define TENSORFLOW_TSL_PLATFORM_MUTEX_H_

#include <chrono>   // NOLINT
#include <cstdint>  // NOLINT
// for std::try_to_lock_t and std::cv_status
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {

enum ConditionResult { kCond_Timeout, kCond_MaybeNotified };
enum LinkerInitialized { LINKER_INITIALIZED };

class condition_variable;
using Condition = absl::Condition;

// Mimic std::mutex + C++17's shared_mutex, adding a LinkerInitialized
// constructor interface.  This type is as fast as mutex, but is also a shared
// lock, and provides conditional critical sections (via Await()), as an
// alternative to condition variables.
class TF_LOCKABLE ABSL_DEPRECATED("Use absl::Mutex instead.") mutex {
 public:
  mutex();
  // The default implementation of the underlying mutex is safe to use after
  // the linker initialization to zero.
  explicit constexpr mutex(LinkerInitialized x) : mu_(absl::kConstInit) {}

  void lock() TF_EXCLUSIVE_LOCK_FUNCTION();
  bool try_lock() TF_EXCLUSIVE_TRYLOCK_FUNCTION(true);
  void unlock() TF_UNLOCK_FUNCTION();
  void assert_held() const TF_ASSERT_EXCLUSIVE_LOCK();

  void lock_shared() TF_SHARED_LOCK_FUNCTION();
  bool try_lock_shared() TF_SHARED_TRYLOCK_FUNCTION(true);
  void unlock_shared() TF_UNLOCK_FUNCTION();
  void assert_held_shared() const TF_ASSERT_SHARED_LOCK();

  // -------
  // Conditional critical sections.
  // These represent an alternative to condition variables that is easier to
  // use.  The predicate must be encapsulated in a function (via Condition),
  // but there is no need to use a while-loop, and no need to signal the
  // condition.  Example:  suppose "mu" protects "counter"; we wish one thread
  // to wait until counter is decremented to zero by another thread.
  //   // Predicate expressed as a function:
  //   static bool IntIsZero(int* pi) { return *pi == 0; }
  //
  //   // Waiter:
  //   mu.lock();
  //   mu.Await(Condition(&IntIsZero, &counter));   // no loop needed
  //   // lock is held and counter==0...
  //   mu.unlock();
  //
  //   // Decrementer:
  //   mu.lock();
  //   counter--;
  //   mu.unlock();    // no need to signal; mutex will check condition
  //
  // A mutex may be used with condition variables and conditional critical
  // sections at the same time.  Conditional critical sections are easier to
  // use, but if there are multiple conditions that are simultaneously false,
  // condition variables may be faster.

  // Unlock *this and wait until cond.Eval() is true, then atomically reacquire
  // *this in the same mode in which it was previously held and return.
  void Await(const Condition& cond);

  // Unlock *this and wait until either cond.Eval is true, or abs_deadline_ns
  // has been reached, then atomically reacquire *this in the same mode in
  // which it was previously held, and return whether cond.Eval() is true.
  // See tsl/tsl/platform/env_time.h for the time interface.
  bool AwaitWithDeadline(const Condition& cond, uint64_t abs_deadline_ns);
  // -------

 private:
  friend class condition_variable;
  absl::Mutex mu_;
};

// Mimic a subset of the std::unique_lock<tsl::mutex> functionality.
class TF_SCOPED_LOCKABLE mutex_lock {
 public:
  typedef ::tsl::mutex mutex_type;

  explicit mutex_lock(mutex_type& mu) TF_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(&mu) {
    mu_->lock();
  }

  mutex_lock(mutex_type& mu, std::try_to_lock_t) TF_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(&mu) {
    if (!mu.try_lock()) {
      mu_ = nullptr;
    }
  }

  // Manually nulls out the source to prevent double-free.
  // (std::move does not null the source pointer by default.)
  mutex_lock(mutex_lock&& ml) noexcept TF_EXCLUSIVE_LOCK_FUNCTION(ml.mu_)
      : mu_(ml.mu_) {
    ml.mu_ = nullptr;
  }
  ~mutex_lock() TF_UNLOCK_FUNCTION() {
    if (mu_ != nullptr) {
      mu_->unlock();
    }
  }
  mutex_type* mutex() { return mu_; }

  explicit operator bool() const { return mu_ != nullptr; }

 private:
  mutex_type* mu_;
};

// Catch bug where variable name is omitted, e.g. mutex_lock (mu);
#define mutex_lock(x) static_assert(0, "mutex_lock_decl_missing_var_name");

// Mimic a subset of the std::shared_lock<tsl::mutex> functionality.
// Name chosen to minimize conflicts with the tf_shared_lock macro, below.
class TF_SCOPED_LOCKABLE tf_shared_lock {
 public:
  typedef ::tsl::mutex mutex_type;

  explicit tf_shared_lock(mutex_type& mu) TF_SHARED_LOCK_FUNCTION(mu)
      : mu_(&mu) {
    mu_->lock_shared();
  }

  tf_shared_lock(mutex_type& mu, std::try_to_lock_t) TF_SHARED_LOCK_FUNCTION(mu)
      : mu_(&mu) {
    if (!mu.try_lock_shared()) {
      mu_ = nullptr;
    }
  }

  // Manually nulls out the source to prevent double-free.
  // (std::move does not null the source pointer by default.)
  tf_shared_lock(tf_shared_lock&& ml) noexcept TF_SHARED_LOCK_FUNCTION(ml.mu_)
      : mu_(ml.mu_) {
    ml.mu_ = nullptr;
  }
  ~tf_shared_lock() TF_UNLOCK_FUNCTION() {
    if (mu_ != nullptr) {
      mu_->unlock_shared();
    }
  }
  mutex_type* mutex() { return mu_; }

  explicit operator bool() const { return mu_ != nullptr; }

 private:
  mutex_type* mu_;
};

// Catch bug where variable name is omitted, e.g. tf_shared_lock (mu);
#define tf_shared_lock(x) \
  static_assert(0, "tf_shared_lock_decl_missing_var_name");

// Mimic std::condition_variable.
class ABSL_DEPRECATED("Use absl::CondVar instead.") condition_variable {
 public:
  condition_variable();

  void wait(mutex_lock& lock);

  template <class Predicate>
  void wait(mutex_lock& lock, Predicate stop_waiting) {
    while (!stop_waiting()) {
      wait(lock);
    }
  }

  template <class Rep, class Period>
  std::cv_status wait_for(mutex_lock& lock,
                          std::chrono::duration<Rep, Period> dur);
  void notify_one();
  void notify_all();

 private:
  friend ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                             condition_variable* cv,
                                             int64_t ms);
  absl::CondVar cv_;
};

// Like "cv->wait(*mu)", except that it only waits for up to "ms" milliseconds.
//
// Returns kCond_Timeout if the timeout expired without this
// thread noticing a signal on the condition variable.  Otherwise may
// return either kCond_Timeout or kCond_MaybeNotified
inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                           condition_variable* cv, int64_t ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}

// ------------------------------------------------------------
// Implementation details follow.   Clients should ignore them.

inline mutex::mutex() = default;

inline void mutex::lock() TF_EXCLUSIVE_LOCK_FUNCTION() { mu_.Lock(); }

inline bool mutex::try_lock() TF_EXCLUSIVE_TRYLOCK_FUNCTION(true) {
  return mu_.TryLock();
};

inline void mutex::unlock() TF_UNLOCK_FUNCTION() { mu_.Unlock(); }

inline void mutex::assert_held() const TF_ASSERT_EXCLUSIVE_LOCK() {
  mu_.AssertHeld();
}

inline void mutex::lock_shared() TF_SHARED_LOCK_FUNCTION() { mu_.ReaderLock(); }

inline bool mutex::try_lock_shared() TF_SHARED_TRYLOCK_FUNCTION(true) {
  return mu_.ReaderTryLock();
}

inline void mutex::unlock_shared() TF_UNLOCK_FUNCTION() { mu_.ReaderUnlock(); }

inline void mutex::assert_held_shared() const TF_ASSERT_SHARED_LOCK() {
  mu_.AssertReaderHeld();
}

inline void mutex::Await(const Condition& cond) { mu_.Await(cond); }

inline bool mutex::AwaitWithDeadline(const Condition& cond,
                                     uint64_t abs_deadline_ns) {
  return mu_.AwaitWithDeadline(cond, absl::FromUnixNanos(abs_deadline_ns));
}

inline condition_variable::condition_variable() = default;

inline void condition_variable::wait(mutex_lock& lock) {
  cv_.Wait(&lock.mutex()->mu_);
}

inline void condition_variable::notify_one() { cv_.Signal(); }

inline void condition_variable::notify_all() { cv_.SignalAll(); }

template <class Rep, class Period>
std::cv_status condition_variable::wait_for(
    mutex_lock& lock, std::chrono::duration<Rep, Period> dur) {
  bool r = cv_.WaitWithTimeout(&lock.mutex()->mu_, ::absl::FromChrono(dur));
  return r ? std::cv_status::timeout : std::cv_status::no_timeout;
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_MUTEX_H_
