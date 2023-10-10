/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_REFFED_STATUS_CALLBACK_H_
#define TENSORFLOW_CORE_UTIL_REFFED_STATUS_CALLBACK_H_

#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// The ReffedStatusCallback is a refcounted object that accepts a
// StatusCallback.  When it is destroyed (its refcount goes to 0), the
// StatusCallback is called with the first non-OK status passed to
// UpdateStatus(), or OkStatus() if no non-OK status was set.
class ReffedStatusCallback : public core::RefCounted {
 public:
  explicit ReffedStatusCallback(StatusCallback done) : done_(std::move(done)) {}

  void UpdateStatus(const Status& s) {
    mutex_lock lock(mu_);
    status_group_.Update(s);
  }

  bool ok() {
    tf_shared_lock lock(mu_);
    return status_group_.ok();
  }

  // Returns a copy of the current status.
  Status status() {
    tf_shared_lock lock(mu_);
    return status_group_.as_summary_status();
  }

  ~ReffedStatusCallback() override { done_(status_group_.as_summary_status()); }

 private:
  StatusCallback done_;
  mutex mu_;
  StatusGroup status_group_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_REFFED_STATUS_CALLBACK_H_
