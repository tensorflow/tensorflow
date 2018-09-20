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
#include <chrono>
#include <condition_variable>
#include "nsync_cv.h"
#include "nsync_mu.h"

namespace tensorflow {

// Check that the external_mu_space struct used to reserve space for the mutex
// in tensorflow::mutex is big enough.
static_assert(sizeof(nsync::nsync_mu) <= sizeof(mutex::external_mu_space),
              "tensorflow::mutex::external_mu_space needs to be bigger");

// Cast a pointer to mutex::external_mu_space to a pointer to the mutex mutex
// representation.  This is done so that the header files for nsync_mu do not
// need to be included in every file that uses tensorflow's mutex.
static inline nsync::nsync_mu *mu_cast(mutex::external_mu_space *mu) {
  return reinterpret_cast<nsync::nsync_mu *>(mu);
}

mutex::mutex() { nsync::nsync_mu_init(mu_cast(&mu_)); }

void mutex::lock() { nsync::nsync_mu_lock(mu_cast(&mu_)); }

bool mutex::try_lock() { return nsync::nsync_mu_trylock(mu_cast(&mu_)) != 0; };

void mutex::unlock() { nsync::nsync_mu_unlock(mu_cast(&mu_)); }

void mutex::lock_shared() { nsync::nsync_mu_rlock(mu_cast(&mu_)); }

bool mutex::try_lock_shared() {
  return nsync::nsync_mu_rtrylock(mu_cast(&mu_)) != 0;
};

void mutex::unlock_shared() { nsync::nsync_mu_runlock(mu_cast(&mu_)); }

// Check that the external_cv_space struct used to reserve space for the
// condition variable in tensorflow::condition_variable is big enough.
static_assert(
    sizeof(nsync::nsync_cv) <= sizeof(condition_variable::external_cv_space),
    "tensorflow::condition_variable::external_cv_space needs to be bigger");

// Cast a pointer to mutex::external_cv_space to a pointer to the condition
// variable representation.  This is done so that the header files for nsync_mu
// do not need to be included in every file that uses tensorflow's
// condition_variable.
static inline nsync::nsync_cv *cv_cast(
    condition_variable::external_cv_space *cv) {
  return reinterpret_cast<nsync::nsync_cv *>(cv);
}

condition_variable::condition_variable() {
  nsync::nsync_cv_init(cv_cast(&cv_));
}

void condition_variable::wait(mutex_lock &lock) {
  nsync::nsync_cv_wait(cv_cast(&cv_), mu_cast(&lock.mutex()->mu_));
}

std::cv_status condition_variable::wait_until_system_clock(
    mutex_lock &lock,
    const std::chrono::system_clock::time_point timeout_time) {
  int r = nsync::nsync_cv_wait_with_deadline(
      cv_cast(&cv_), mu_cast(&lock.mutex()->mu_), timeout_time, nullptr);
  return r ? std::cv_status::timeout : std::cv_status::no_timeout;
}

void condition_variable::notify_one() { nsync::nsync_cv_signal(cv_cast(&cv_)); }

void condition_variable::notify_all() {
  nsync::nsync_cv_broadcast(cv_cast(&cv_));
}

}  // namespace tensorflow
