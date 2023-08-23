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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_CHANNEL_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_CHANNEL_H_

#include <queue>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace tfrt_stub {

// An unbounded queue for communicating between threads. This class is
// thread-safe.
template <typename T>
class UnboundedChannel {
 public:
  absl::Status Write(T value) {
    absl::MutexLock lock(&mu_);

    if (closed_) {
      return absl::InternalError(
          "Failed to write to the UnboundedChannel that is closed.");
    }

    channel_.push(std::move(value));

    return absl::OkStatus();
  }

  bool Read(T& value) {
    absl::MutexLock lock(&mu_);

    mu_.Await(absl::Condition(
        +[](UnboundedChannel* channel) ABSL_SHARED_LOCKS_REQUIRED(mu_) {
          return !channel->channel_.empty() || channel->closed_;
        },
        this));

    if (!channel_.empty()) {
      value = std::move(channel_.front());
      channel_.pop();
      return true;
    }

    // If channel_ is empty, then it must be closed at this point.
    DCHECK(closed_);
    return false;
  }

  void Close() {
    absl::MutexLock lock(&mu_);
    closed_ = true;
  }

 private:
  absl::Mutex mu_;
  std::queue<T> channel_ ABSL_GUARDED_BY(mu_);
  bool closed_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_CHANNEL_H_
