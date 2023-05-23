/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NON_ATOMICALLY_UPGRADEABLE_RW_LOCK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NON_ATOMICALLY_UPGRADEABLE_RW_LOCK_H_

#include <memory>
#include <variant>

#include "absl/synchronization/mutex.h"

namespace xla {
namespace gpu {

// Augments absl::ReaderMutexLock with a poor man's upgrade/downgrade pair using
// RAII. Instead of a true upgrade (or downgrade), we simply drop the read
// (write) lock and then reacquire it as a write (read) lock.
class ABSL_SCOPED_LOCKABLE NonAtomicallyUpgradeableRWLock {
 public:
  explicit NonAtomicallyUpgradeableRWLock(absl::Mutex* mu)
      ABSL_SHARED_LOCK_FUNCTION(mu)
      : mu_(mu), is_reader_(true) {
    mu_->ReaderLock();
  }

  NonAtomicallyUpgradeableRWLock(const NonAtomicallyUpgradeableRWLock&) =
      delete;
  NonAtomicallyUpgradeableRWLock(NonAtomicallyUpgradeableRWLock&&) = delete;
  NonAtomicallyUpgradeableRWLock& operator=(
      const NonAtomicallyUpgradeableRWLock&) = delete;
  NonAtomicallyUpgradeableRWLock& operator=(NonAtomicallyUpgradeableRWLock&&) =
      delete;

  ~NonAtomicallyUpgradeableRWLock() ABSL_UNLOCK_FUNCTION() {
    if (is_reader_) {
      mu_->ReaderUnlock();
    } else {
      mu_->WriterUnlock();
    }
  }

  // Upgrade and downgrade the reader lock via RAII.
  class ABSL_SCOPED_LOCKABLE WriterLock {
   public:
    explicit WriterLock(NonAtomicallyUpgradeableRWLock* parent)
        ABSL_EXCLUSIVE_LOCK_FUNCTION(parent->mu_)
        : parent_(parent) {
      assert(parent_->is_reader_);
      parent_->mu_->ReaderUnlock();
      parent_->mu_->WriterLock();
      parent_->is_reader_ = false;
    }

    WriterLock(const WriterLock&) = delete;
    WriterLock(WriterLock&&) = delete;
    WriterLock& operator=(const WriterLock&) = delete;
    WriterLock& operator=(WriterLock&&) = delete;

    ~WriterLock() ABSL_UNLOCK_FUNCTION() {
      parent_->mu_->WriterUnlock();
      parent_->mu_->ReaderLock();
      parent_->is_reader_ = true;
    }

   private:
    NonAtomicallyUpgradeableRWLock* parent_;
  };

  // Update the reader lock to a writer lock. The function is invalid if the
  // lock is already upgraded.
  WriterLock UpgradeToWriterMutexLock() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return WriterLock(this);
  }

 private:
  absl::Mutex* const mu_;
  bool is_reader_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NON_ATOMICALLY_UPGRADEABLE_RW_LOCK_H_
