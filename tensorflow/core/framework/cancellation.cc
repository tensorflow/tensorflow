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

#include "tensorflow/core/framework/cancellation.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

const CancellationToken CancellationManager::kInvalidToken = -1;

CancellationManager::CancellationManager()
    : is_cancelling_(false),
      is_cancelled_(false),
      next_cancellation_token_(0) {}

void CancellationManager::StartCancel() {
  gtl::FlatMap<CancellationToken, CancelCallback> callbacks_to_run;
  {
    mutex_lock l(mu_);
    if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
      return;
    }
    is_cancelling_ = true;
    std::swap(callbacks_, callbacks_to_run);
  }
  // We call these callbacks without holding mu_, so that concurrent
  // calls to DeregisterCallback, which can happen asynchronously, do
  // not block. The callbacks remain valid because any concurrent call
  // to DeregisterCallback will block until the
  // cancelled_notification_ is notified.
  for (auto key_and_value : callbacks_to_run) {
    key_and_value.second();
  }
  {
    mutex_lock l(mu_);
    is_cancelling_ = false;
    is_cancelled_.store(true, std::memory_order_release);
  }
  cancelled_notification_.Notify();
}

CancellationToken CancellationManager::get_cancellation_token() {
  mutex_lock l(mu_);
  return next_cancellation_token_++;
}

bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
  mutex_lock l(mu_);
  CHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";
  bool should_register = !is_cancelled_ && !is_cancelling_;
  if (should_register) {
    std::swap(callbacks_[token], callback);
  }
  return should_register;
}

bool CancellationManager::DeregisterCallback(CancellationToken token) {
  mu_.lock();
  if (is_cancelled_) {
    mu_.unlock();
    return false;
  } else if (is_cancelling_) {
    mu_.unlock();
    // Wait for all of the cancellation callbacks to be called. This
    // wait ensures that the caller of DeregisterCallback does not
    // return immediately and free objects that may be used in the
    // execution of any currently pending callbacks in StartCancel.
    cancelled_notification_.WaitForNotification();
    return false;
  } else {
    callbacks_.erase(token);
    mu_.unlock();
    return true;
  }
}

CancellationManager::~CancellationManager() {
  if (!callbacks_.empty()) {
    StartCancel();
  }
}

}  // end namespace tensorflow
