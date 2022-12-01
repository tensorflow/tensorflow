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

#include "tensorflow/compiler/xla/python/ifrt/future.h"

#include <atomic>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace ifrt {

Future<Status> JoinFutures(absl::Span<Future<Status>> futures) {
  if (futures.empty()) {
    return Future<Status>(OkStatus());
  } else if (futures.size() == 1) {
    return futures.front();
  }
  // State shared by `PjRtFuture` onready callbacks.
  struct CombinedStatus {
    explicit CombinedStatus(int initial_count)
        : count(initial_count), promise(Future<Status>::CreatePromise()) {}
    std::atomic<int> count;
    absl::Mutex mu;
    Status status ABSL_GUARDED_BY(&mu);
    Promise<Status> promise;
  };
  auto combined_status = std::make_shared<CombinedStatus>(futures.size());
  Future<Status> future(combined_status->promise);
  for (auto& fut : futures) {
    fut.OnReady([combined_status](Status s) {
      if (!s.ok()) {
        absl::MutexLock lock(&combined_status->mu);
        combined_status->status.Update(std::move(s));
      }
      const int pre_dec_count =
          combined_status->count.fetch_add(-1, std::memory_order_acq_rel);
      CHECK_GE(pre_dec_count, 1);
      if (pre_dec_count == 1) {
        absl::MutexLock lock(&combined_status->mu);
        combined_status->promise.Set(std::move(combined_status->status));
      }
    });
  }
  return future;
}

}  // namespace ifrt
}  // namespace xla
