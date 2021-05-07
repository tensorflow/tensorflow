/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
const Status& ThreadSafeStatus::status() const& {
  tf_shared_lock lock(mutex_);
  return status_;
}

Status ThreadSafeStatus::status() && {
  tf_shared_lock lock(mutex_);
  return std::move(status_);
}

void ThreadSafeStatus::Update(const Status& new_status) {
  if (new_status.ok()) {
    return;
  }

  mutex_lock lock(mutex_);
  status_.Update(new_status);
}

void ThreadSafeStatus::Update(Status&& new_status) {
  if (new_status.ok()) {
    return;
  }

  mutex_lock lock(mutex_);
  status_.Update(std::forward<Status>(new_status));
}
}  // namespace tensorflow
