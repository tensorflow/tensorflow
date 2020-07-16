/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/mutex.h"

#include <time.h>

#include "nsync_cv.h"       // NOLINT
#include "nsync_mu.h"       // NOLINT
#include "nsync_mu_wait.h"  // NOLINT
#include "nsync_time.h"     // NOLINT

namespace tensorflow {

// Check that the MuData struct used to reserve space for the mutex
// in tensorflow::mutex is big enough.
static_assert(sizeof(nsync::nsync_mu) <= sizeof(internal::MuData),
              "tensorflow::internal::MuData needs to be bigger");

// Cast a pointer to internal::MuData to a pointer to the mutex
// representation.  This is done so that the header files for nsync_mu do not
// need to be included in every file that uses tensorflow's mutex.
static inline nsync::nsync_mu *mu_cast(internal::MuData *mu) {
  return reinterpret_cast<nsync::nsync_mu *>(mu);
}

mutex::mutex() { nsync::nsync_mu_init(mu_cast(&mu_)); }

mutex::mutex(LinkerInitialized x) {}

void mutex::lock() { nsync::nsync_mu_lock(mu_cast(&mu_)); }

bool mutex::try_lock() { return nsync::nsync_mu_trylock(mu_cast(&mu_)) != 0; };

void mutex::unlock() { nsync::nsync_mu_unlock(mu_cast(&mu_)); }

void mutex::lock_shared() { nsync::nsync_mu_rlock(mu_cast(&mu_)); }

bool mutex::try_lock_shared() {
  return nsync::nsync_mu_rtrylock(mu_cast(&mu_)) != 0;
};

void mutex::unlock_shared() { nsync::nsync_mu_runlock(mu_cast(&mu_)); }

// A callback suitable for nsync_mu_wait() that calls Condition::Eval().
static int EvaluateCondition(const void *vcond) {
  return static_cast<int>(static_cast<const Condition *>(vcond)->Eval());
}

void mutex::Await(const Condition &cond) {
  nsync::nsync_mu_wait(mu_cast(&mu_), &EvaluateCondition, &cond, nullptr);
}

bool mutex::AwaitWithDeadline(const Condition &cond, uint64 abs_deadline_ns) {
  time_t seconds = abs_deadline_ns / (1000 * 1000 * 1000);
  nsync::nsync_time abs_time = nsync::nsync_time_s_ns(
      seconds, abs_deadline_ns - seconds * (1000 * 1000 * 1000));
  return nsync::nsync_mu_wait_with_deadline(mu_cast(&mu_), &EvaluateCondition,
                                            &cond, nullptr, abs_time,
                                            nullptr) == 0;
}

// Check that the CVData struct used to reserve space for the
// condition variable in tensorflow::condition_variable is big enough.
static_assert(sizeof(nsync::nsync_cv) <= sizeof(internal::CVData),
              "tensorflow::internal::CVData needs to be bigger");

// Cast a pointer to internal::CVData to a pointer to the condition
// variable representation.  This is done so that the header files for nsync_cv
// do not need to be included in every file that uses tensorflow's
// condition_variable.
static inline nsync::nsync_cv *cv_cast(internal::CVData *cv) {
  return reinterpret_cast<nsync::nsync_cv *>(cv);
}

condition_variable::condition_variable() {
  nsync::nsync_cv_init(cv_cast(&cv_));
}

void condition_variable::wait(mutex_lock &lock) {
  nsync::nsync_cv_wait(cv_cast(&cv_), mu_cast(&lock.mutex()->mu_));
}

void condition_variable::notify_one() { nsync::nsync_cv_signal(cv_cast(&cv_)); }

void condition_variable::notify_all() {
  nsync::nsync_cv_broadcast(cv_cast(&cv_));
}

namespace internal {
std::cv_status wait_until_system_clock(
    CVData *cv_data, MuData *mu_data,
    const std::chrono::system_clock::time_point timeout_time) {
  int r = nsync::nsync_cv_wait_with_deadline(cv_cast(cv_data), mu_cast(mu_data),
                                             timeout_time, nullptr);
  return r ? std::cv_status::timeout : std::cv_status::no_timeout;
}
}  // namespace internal

}  // namespace tensorflow
