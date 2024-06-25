/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_future.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace {
struct State {
  explicit State(int32_t size)
      : pending_count(size), promise(PjRtFuture<>::CreatePromise()) {}

  std::atomic<int32_t> pending_count;
  PjRtFuture<>::Promise promise;

  absl::Mutex mu;
  absl::Status status ABSL_GUARDED_BY(&mu);
};
}  // namespace

PjRtFuture<> JoinFutures(absl::Span<const PjRtFuture<>> futures) {
  if (futures.empty()) {
    return PjRtFuture<>(absl::OkStatus());
  } else if (futures.size() == 1) {
    return futures.front();
  }

  auto state = std::make_shared<State>(futures.size());

  for (const PjRtFuture<>& future : futures) {
    future.OnReady([state](absl::Status status) {
      if (!status.ok()) {
        absl::MutexLock lock(&state->mu);
        state->status.Update(status);
      }

      const int pending_count =
          state->pending_count.fetch_sub(1, std::memory_order_acq_rel);
      CHECK_GE(pending_count, 1) << "Pending count can't drop below 0";

      if (pending_count == 1) {
        absl::MutexLock lock(&state->mu);
        state->promise.Set(std::move(state->status));
      }
    });
  }

  return PjRtFuture<>(state->promise);
}

}  // namespace xla
