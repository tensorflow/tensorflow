/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_barrier_proxy.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace tensorflow {

std::pair<absl::Status, bool> BarrierProxy::Wait() {
  mutex_lock l(mu_);
  if (status_set_) {
    return std::make_pair(
        absl::FailedPreconditionError(absl::StrCat(
            "The barrier has already passed or timed out. key=", key_)),
        false);
  }
  if (num_entered_ >= num_local_threads_) {
    return std::make_pair(absl::FailedPreconditionError(absl::StrCat(
                              "Wait() called too many (>", num_local_threads_,
                              ") times. key=", key_)),
                          false);
  }
  // Now that `Wait` has passed pre-condition check, the thread has entered the
  // barrier.
  ++num_entered_;
  ++num_to_exit_;
  VLOG(1) << "BarrierProxy " << key_ << " enter: num_entered_=" << num_entered_
          << ", num_to_exit_=" << num_to_exit_;

  if (num_entered_ == num_local_threads_) {
    // Now that all threads are waiting, starts waiting at the global barrier.
    if (tasks_.size() != 1) {
      tsl::profiler::TraceMe traceme("BarrierProxy::Wait::WaitAtBarrier");
      // TODO(b/198475014) the barrier status will be stored in memory forever.
      // We should have a mechanism to remove it after it has been passed.
      status_ = agent_->WaitAtBarrier(key_, timeout_, tasks_);
    } else {
      status_ = absl::OkStatus();
    }
    status_set_ = true;
    cv_.notify_all();
  } else if (WaitForMilliseconds(&l, &cv_, timeout_ / absl::Milliseconds(1)) ==
             kCond_Timeout) {
    if (!status_set_) {
      if (tasks_.size() != 1) {
        // Cancels the global barrier. Okay to ignore errors because remote
        // tasks' WaitAtBarrier will time out anyway if the barrier is not
        // cancelled for any reason.
        agent_->CancelBarrier(key_).IgnoreError();
      }
      status_ = absl::DeadlineExceededError(
          absl::StrCat("BarrierProxy timeout: key=", key_));
      status_set_ = true;
      cv_.notify_all();
    }
  } else {
    CHECK(status_set_);  // Crash OK
  }
  // The thread is exiting the barrier.
  --num_to_exit_;
  VLOG(1) << "BarrierProxy " << key_ << " enter: num_entered_=" << num_entered_
          << ", num_to_exit=" << num_to_exit_;
  return std::make_pair(status_, num_to_exit_ == 0);
}

size_t BarrierProxyManager::size() const {
  mutex_lock l(mu_);
  return barriers_.size();
}

absl::Status BarrierProxyManager::Wait(
    tsl::CoordinationServiceAgent* agent,
    const std::vector<CoordinatedTask>& tasks, int num_local_threads,
    absl::string_view key, absl::Duration timeout) {
  // Only one device, no need to wait.
  if (tasks.size() == 1 && num_local_threads <= 1) return absl::OkStatus();

  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        "BarrierProxyManager::Wait",
        {
            {"num_tasks", tasks.size()},
            {"num_local_threads", num_local_threads},
        });
  });

  std::shared_ptr<BarrierProxy> barrier;
  {
    mutex_lock l(mu_);
    auto [iter, inserted] = barriers_.try_emplace(key);
    if (inserted) {
      iter->second = std::make_shared<BarrierProxy>(
          agent, tasks, num_local_threads, key, timeout);
      VLOG(1) << "BarrierProxy key=" << key << " created.";
    }
    barrier = iter->second;
  }
  CHECK(barrier);  // Crash OK
  auto [status, last_exit] = barrier->Wait();
  if (last_exit) {
    mutex_lock l(mu_);
    barriers_.erase(key);
    VLOG(1) << "BarrierProxy key=" << key << " removed.";
  }
  return status;
}

}  // namespace tensorflow
