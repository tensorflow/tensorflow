/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_SEMAPHORE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_SEMAPHORE_H_

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class Semaphore {
 public:
  explicit Semaphore(int64 capacity);

  // Acquires `amount` units. Blocks until `amount` units are available.
  void Acquire(int64 amount);

  // Returns `amount` units to the semaphore.
  void Release(int64 amount);

  class ScopedReservation {
   public:
    ScopedReservation(Semaphore* semaphore, int64 amount)
        : semaphore_(semaphore), amount_(amount) {}
    ~ScopedReservation();

    ScopedReservation(const ScopedReservation&) = delete;
    ScopedReservation(ScopedReservation&& other) noexcept;
    ScopedReservation& operator=(const ScopedReservation&) = delete;
    ScopedReservation& operator=(ScopedReservation&& other) noexcept;

   private:
    Semaphore* semaphore_;
    int64 amount_;
  };
  // RAII version of Acquire. Releases the reservation when the
  // ScopedReservation is destroyed.
  ScopedReservation ScopedAcquire(int64 amount);

 private:
  struct CanAcquireArgs {
    Semaphore* semaphore;
    int64 amount;
  };
  static bool CanAcquire(CanAcquireArgs* args)
      EXCLUSIVE_LOCKS_REQUIRED(args->semaphore->mu_);

  absl::Mutex mu_;
  int64 value_ GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SEMAPHORE_H_
