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

#ifndef TENSORFLOW_CORE_PLATFORM_MUTEX_H_
#define TENSORFLOW_CORE_PLATFORM_MUTEX_H_

#include <chrono>  // NOLINT
// for std::try_to_lock_t and std::cv_status
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

// Include appropriate platform-dependent implementation details of mutex etc.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/mutex_data.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/default/mutex_data.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace tensorflow {

enum ConditionResult { kCond_Timeout, kCond_MaybeNotified };
enum LinkerInitialized { LINKER_INITIALIZED };

class condition_variable;
class Condition;

// Mimic std::mutex + C++17's shared_mutex, adding a LinkerInitialized
// constructor interface.  This type is as fast as mutex, but is also a shared
// lock, and provides conditional critical sections (via Await()), as an
// alternative to condition variables.
class TF_LOCKABLE mutex {
 public:
  mutex();
  // The default implementation of the underlying mutex is safe to use after
  // the linker initialization to zero.
  explicit mutex(LinkerInitialized x);

  void lock() TF_EXCLUSIVE_LOCK_FUNCTION();
  bool try_lock() TF_EXCLUSIVE_TRYLOCK_FUNCTION(true);
  void unlock() TF_UNLOCK_FUNCTION();

  void lock_shared() TF_SHARED_LOCK_FUNCTION();
  bool try_lock_shared() TF_SHARED_TRYLOCK_FUNCTION(true);
  void unlock_shared() TF_UNLOCK_FUNCTION();

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
  // See tensorflow/core/platform/env_time.h for the time interface.
  bool AwaitWithDeadline(const Condition& cond, uint64 abs_deadline_ns);
  // -------

 private:
  friend class condition_variable;
  internal::MuData mu_;
};

// A Condition represents a predicate on state protected by a mutex.  The
// function must have no side-effects on that state.  When passed to
// mutex::Await(), the function will be called with the mutex held.  It may be
// called:
// - any number of times;
// - by any thread using the mutex; and/or
// - with the mutex held in any mode (read or write).
// If you must use a lambda, prefix the lambda with +, and capture no variables.
// For example:  Condition(+[](int *pi)->bool { return *pi == 0; }, &i)
class Condition {
 public:
  template <typename T>
  Condition(bool (*func)(T* arg), T* arg);  // Value is (*func)(arg)
  template <typename T>
  Condition(T* obj, bool (T::*method)());  // Value is obj->*method()
  template <typename T>
  Condition(T* obj, bool (T::*method)() const);  // Value is obj->*method()
  explicit Condition(const bool* flag);          // Value is *flag

  // Return the value of the predicate represented by this Condition.
  bool Eval() const { return (*this->eval_)(this); }

 private:
  bool (*eval_)(const Condition*);  // CallFunction, CallMethod, or, ReturnBool
  bool (*function_)(void*);         // predicate of form (*function_)(arg_)
  bool (Condition::*method_)();     // predicate of form arg_->method_()
  void* arg_;
  Condition();
  // The following functions can be pointed to by the eval_ field.
  template <typename T>
  static bool CallFunction(const Condition* cond);  // call function_
  template <typename T>
  static bool CallMethod(const Condition* cond);  // call method_
  static bool ReturnBool(const Condition* cond);  // access *(bool *)arg_
};

// Mimic a subset of the std::unique_lock<tensorflow::mutex> functionality.
class TF_SCOPED_LOCKABLE mutex_lock {
 public:
  typedef ::tensorflow::mutex mutex_type;

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

// Mimic a subset of the std::shared_lock<tensorflow::mutex> functionality.
// Name chosen to minimize conflicts with the tf_shared_lock macro, below.
class TF_SCOPED_LOCKABLE tf_shared_lock {
 public:
  typedef ::tensorflow::mutex mutex_type;

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
class condition_variable {
 public:
  condition_variable();

  void wait(mutex_lock& lock);
  template <class Rep, class Period>
  std::cv_status wait_for(mutex_lock& lock,
                          std::chrono::duration<Rep, Period> dur);
  void notify_one();
  void notify_all();

 private:
  friend ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                             condition_variable* cv, int64 ms);
  internal::CVData cv_;
};

// Like "cv->wait(*mu)", except that it only waits for up to "ms" milliseconds.
//
// Returns kCond_Timeout if the timeout expired without this
// thread noticing a signal on the condition variable.  Otherwise may
// return either kCond_Timeout or kCond_MaybeNotified
inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                           condition_variable* cv, int64 ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}

// ------------------------------------------------------------
// Implementation details follow.   Clients should ignore them.

// private static
template <typename T>
inline bool Condition::CallFunction(const Condition* cond) {
  bool (*fn)(T*) = reinterpret_cast<bool (*)(T*)>(cond->function_);
  return (*fn)(static_cast<T*>(cond->arg_));
}

template <typename T>
inline Condition::Condition(bool (*func)(T*), T* arg)
    : eval_(&CallFunction<T>),
      function_(reinterpret_cast<bool (*)(void*)>(func)),
      method_(nullptr),
      arg_(const_cast<void*>(static_cast<const void*>(arg))) {}

// private static
template <typename T>
inline bool Condition::CallMethod(const Condition* cond) {
  bool (T::*m)() = reinterpret_cast<bool (T::*)()>(cond->method_);
  return (static_cast<T*>(cond->arg_)->*m)();
}

template <typename T>
inline Condition::Condition(T* obj, bool (T::*method)())
    : eval_(&CallMethod<T>),
      function_(nullptr),
      method_(reinterpret_cast<bool (Condition::*)()>(method)),
      arg_(const_cast<void*>(static_cast<const void*>(obj))) {}

template <typename T>
inline Condition::Condition(T* obj, bool (T::*method)() const)
    : eval_(&CallMethod<T>),
      function_(nullptr),
      method_(reinterpret_cast<bool (Condition::*)()>(method)),
      arg_(const_cast<void*>(static_cast<const void*>(obj))) {}

// private static
inline bool Condition::ReturnBool(const Condition* cond) {
  return *static_cast<bool*>(cond->arg_);
}

inline Condition::Condition(const bool* flag)
    : eval_(&ReturnBool),
      function_(nullptr),
      method_(nullptr),
      arg_(const_cast<void*>(static_cast<const void*>(flag))) {}

}  // namespace tensorflow

// Include appropriate platform-dependent implementation details of mutex etc.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/mutex.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/default/mutex.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_MUTEX_H_
