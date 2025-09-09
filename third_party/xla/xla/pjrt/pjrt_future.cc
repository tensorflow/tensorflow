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
#include <tuple>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// Construct an immediately ready promise in the static storage. This avoids
// heap allocation and reference counting operations on a hot path.
static tsl::internal::AsyncValueStorage<absl::Status> ready_promise_storage;
absl::NoDestructor<tsl::AsyncValueOwningRef<absl::Status>>
    PjRtFuture<>::ready_promise_(
        tsl::MakeAvailableAsyncValueRef<absl::Status>(ready_promise_storage));

namespace {
struct State {
  explicit State(int32_t size) : pending_count(size) {
    std::tie(promise, future) = PjRtFuture<>::MakePromise();
  }

  std::atomic<int32_t> pending_count;
  PjRtFuture<>::Promise promise;
  PjRtFuture<> future;

  absl::Mutex mu;
  absl::Status status ABSL_GUARDED_BY(&mu);
};
}  // namespace

PjRtFuture<> JoinFutures(absl::Span<const PjRtFuture<>> futures) {
  VLOG(2) << "xla::JoinFutures: " << futures.size() << " futures";
  if (futures.empty()) {
    return PjRtFuture<>(absl::OkStatus());
  }
  if (futures.size() == 1) {
    return futures.front();
  }

  auto state = std::make_shared<State>(futures.size());

  for (const PjRtFuture<>& future : futures) {
    future.OnReady([state](absl::Status status) {
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        absl::MutexLock lock(&state->mu);
        if (VLOG_IS_ON(2)) {
          if (!state->status.ok() && status.code() != state->status.code()) {
            VLOG(2) << "Ignoring status " << status
                    << " because first error was " << state->status;
          }
        }
        state->status.Update(status);
      }

      int32_t pending_count =
          state->pending_count.fetch_sub(1, std::memory_order_acq_rel);
      CHECK_GE(pending_count, 1) << "Pending count can't drop below 0";

      if (pending_count == 1) {
        absl::MutexLock lock(&state->mu);
        state->promise.Set(std::move(state->status));
      }
    });
  }

  return std::move(state->future);
}

}  // namespace xla
