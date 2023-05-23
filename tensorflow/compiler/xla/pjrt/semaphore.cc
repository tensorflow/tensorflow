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

#include "tensorflow/compiler/xla/pjrt/semaphore.h"

#include "tensorflow/tsl/platform/logging.h"

namespace xla {

Semaphore::Semaphore(int64_t capacity) : value_(capacity) {
  CHECK_GE(capacity, 0);
}

bool Semaphore::CanAcquire(CanAcquireArgs* args) {
  return args->semaphore->value_ >= args->amount;
}

void Semaphore::Acquire(int64_t amount) {
  CHECK_GE(amount, 0);

  CanAcquireArgs args;
  args.semaphore = this;
  args.amount = amount;

  mu_.LockWhen(absl::Condition(&CanAcquire, &args));
  value_ -= amount;
  mu_.Unlock();
}

void Semaphore::Release(int64_t amount) {
  CHECK_GE(amount, 0);
  absl::MutexLock lock(&mu_);
  value_ += amount;
}

Semaphore::ScopedReservation::~ScopedReservation() {
  if (semaphore_) {
    semaphore_->Release(amount_);
  }
}

Semaphore::ScopedReservation::ScopedReservation(
    ScopedReservation&& other) noexcept {
  semaphore_ = other.semaphore_;
  amount_ = other.amount_;
  other.semaphore_ = nullptr;
}

Semaphore::ScopedReservation& Semaphore::ScopedReservation::operator=(
    ScopedReservation&& other) noexcept {
  semaphore_ = other.semaphore_;
  amount_ = other.amount_;
  other.semaphore_ = nullptr;
  return *this;
}

Semaphore::ScopedReservation Semaphore::ScopedAcquire(int64_t amount) {
  Acquire(amount);
  return ScopedReservation(this, amount);
}

}  // namespace xla
