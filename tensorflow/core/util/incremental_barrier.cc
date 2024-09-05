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

#include "tensorflow/core/util/incremental_barrier.h"

#include <atomic>
#include <functional>
#include <utility>

#include "absl/functional/bind_front.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class InternalIncrementalBarrier {
 public:
  explicit InternalIncrementalBarrier(IncrementalBarrier::DoneCallback callback)
      : left_(1), done_callback_(std::move(callback)) {}

  void operator()() {
    DCHECK_GE(left_.load(std::memory_order_relaxed), 0);

    if (left_.fetch_sub(1, std::memory_order_acq_rel) - 1 == 0) {
      IncrementalBarrier::DoneCallback done_callback =
          std::move(done_callback_);
      delete this;
      done_callback();
    }
  }

  IncrementalBarrier::BarrierCallback Inc() {
    left_.fetch_add(1, std::memory_order_acq_rel);

    // std::bind_front is only available ever since C++20.
    return absl::bind_front(&InternalIncrementalBarrier::operator(), this);
  }

 private:
  std::atomic<int> left_;
  IncrementalBarrier::DoneCallback done_callback_;
};

IncrementalBarrier::IncrementalBarrier(DoneCallback done_callback)
    : internal_barrier_(
          new InternalIncrementalBarrier(std::move(done_callback))) {}

IncrementalBarrier::~IncrementalBarrier() { (*internal_barrier_)(); }

IncrementalBarrier::BarrierCallback IncrementalBarrier::Inc() {
  return internal_barrier_->Inc();
}

}  // namespace tensorflow
