/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_UTIL_FREE_THREADING_MUTEX_H_
#define TENSORFLOW_PYTHON_UTIL_FREE_THREADING_MUTEX_H_

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace py_util {

// An ABI-stable mutex wrapper designed for Python free-threaded (No-GIL)
// builds.
//
// In free-threaded builds, it wraps a real `absl::Mutex` to enforce thread
// safety across parallel C++ extension calls. In standard GIL-enabled builds,
// the locking operations compile down to zero-overhead no-ops because the
// compiler inlines the empty operations.
//
// Crucially, the `absl::Mutex` member is ALWAYS present in the struct,
// ensuring that `FreeThreadingMutex` has the identical struct size (8 bytes on
// 64-bit Linux) and layout in all builds. This completely eliminates any risk
// of ABI layout mismatches or ODR violations even if the header is
// transitively included by external C++ libraries compiled with different
// Python/compiler flags.
class ABSL_LOCKABLE FreeThreadingMutex {
 public:
  FreeThreadingMutex() = default;
  explicit constexpr FreeThreadingMutex(absl::ConstInitType)
      : mutex_(absl::kConstInit) {}

  FreeThreadingMutex(const FreeThreadingMutex&) = delete;
  FreeThreadingMutex& operator=(const FreeThreadingMutex&) = delete;

  void ReaderLock() ABSL_SHARED_LOCK_FUNCTION() {
#ifdef Py_GIL_DISABLED
    mutex_.ReaderLock();
#endif
  }

  void ReaderUnlock() ABSL_UNLOCK_FUNCTION() {
#ifdef Py_GIL_DISABLED
    mutex_.ReaderUnlock();
#endif
  }

  void WriterLock() ABSL_EXCLUSIVE_LOCK_FUNCTION() {
#ifdef Py_GIL_DISABLED
    mutex_.WriterLock();
#endif
  }

  void WriterUnlock() ABSL_UNLOCK_FUNCTION() {
#ifdef Py_GIL_DISABLED
    mutex_.WriterUnlock();
#endif
  }

 private:
  // Always present to ensure ABI struct layout stability.
  absl::Mutex mutex_;
};

// RAII reader lock guard for FreeThreadingMutex.
class ABSL_SCOPED_LOCKABLE FreeThreadingReaderMutexLock {
 public:
  explicit FreeThreadingReaderMutexLock(const FreeThreadingMutex* mu)
      ABSL_SHARED_LOCK_FUNCTION(mu)
      : mu_(const_cast<FreeThreadingMutex*>(mu)) {
    if (mu_) {
      mu_->ReaderLock();
    }
  }

  ~FreeThreadingReaderMutexLock() ABSL_UNLOCK_FUNCTION() {
    if (mu_) {
      mu_->ReaderUnlock();
    }
  }

  FreeThreadingReaderMutexLock(const FreeThreadingReaderMutexLock&) = delete;
  FreeThreadingReaderMutexLock& operator=(const FreeThreadingReaderMutexLock&) =
      delete;

 private:
  FreeThreadingMutex* const mu_;
};

// RAII writer lock guard for FreeThreadingMutex.
class ABSL_SCOPED_LOCKABLE FreeThreadingWriterMutexLock {
 public:
  explicit FreeThreadingWriterMutexLock(FreeThreadingMutex* mu)
      ABSL_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(mu) {
    if (mu_) {
      mu_->WriterLock();
    }
  }

  ~FreeThreadingWriterMutexLock() ABSL_UNLOCK_FUNCTION() {
    if (mu_) {
      mu_->WriterUnlock();
    }
  }

  FreeThreadingWriterMutexLock(const FreeThreadingWriterMutexLock&) = delete;
  FreeThreadingWriterMutexLock& operator=(const FreeThreadingWriterMutexLock&) =
      delete;

 private:
  FreeThreadingMutex* const mu_;
};

}  // namespace py_util
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_FREE_THREADING_MUTEX_H_
