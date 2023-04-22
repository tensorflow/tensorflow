/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/task_remover.h"

#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace {
const int64 kWaitTimeoutUs = 10 * 1000 * 1000;  // 10 seconds.
const int64 kInvalidRound = -1;
}  // namespace

TaskRemover::TaskRemover(int64 num_consumers) : num_consumers_(num_consumers) {}

bool TaskRemover::RequestRemoval(int64 consumer_index, int64 round) {
  mutex_lock l(mu_);
  if (consumers_waiting_.empty()) {
    round_ = round;
  }
  if (round != round_) {
    round_ = kInvalidRound;
    cv_.notify_all();
    return false;
  }
  consumers_waiting_.insert(consumer_index);
  auto cleanup = gtl::MakeCleanup([&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    consumers_waiting_.erase(consumer_index);
  });
  int64 deadline_us = Env::Default()->NowMicros() + kWaitTimeoutUs;
  while (round == round_ && !removed_ &&
         consumers_waiting_.size() < num_consumers_ &&
         Env::Default()->NowMicros() < deadline_us) {
    cv_.wait_for(l, std::chrono::microseconds(deadline_us -
                                              Env::Default()->NowMicros()));
  }
  if (removed_) {
    return true;
  }
  if (consumers_waiting_.size() == num_consumers_) {
    removed_ = true;
    round_ = kInvalidRound;
    cv_.notify_all();
    return true;
  }
  // If we get here it either means timeout was reached, or another consumer
  // requested removal for a different round.
  return false;
}

}  // namespace data
}  // namespace tensorflow
