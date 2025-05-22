/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/distributed_runtime/call_options.h"

#include <utility>

#include "absl/synchronization/mutex.h"

namespace tsl {

CallOptions::CallOptions() = default;

void CallOptions::StartCancel() {
  absl::MutexLock l(&mu_);
  if (cancel_func_ != nullptr) {
    // NOTE: We must call the cancel_func_ with mu_ held. This ensure
    // that ClearCancelCallback() does not race with StartCancel().
    cancel_func_();
    // NOTE: We can clear cancel_func_ if needed.
  }
}

void CallOptions::SetCancelCallback(CancelFunction cancel_func) {
  absl::MutexLock l(&mu_);
  cancel_func_ = std::move(cancel_func);
}

void CallOptions::ClearCancelCallback() {
  absl::MutexLock l(&mu_);
  cancel_func_ = nullptr;
}

int64_t CallOptions::GetTimeout() {
  absl::MutexLock l(&mu_);
  return timeout_in_ms_;
}

void CallOptions::SetTimeout(int64_t ms) {
  absl::MutexLock l(&mu_);
  timeout_in_ms_ = ms;
}

}  // end namespace tsl
