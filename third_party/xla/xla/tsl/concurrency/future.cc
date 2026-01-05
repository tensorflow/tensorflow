/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/concurrency/future.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace tsl {

// Construct an immediately ready promise in the static storage. This avoids
// heap allocation and reference counting operations on a hot path.
static tsl::internal::AsyncValueStorage<absl::Status> ready_promise_storage;
absl::NoDestructor<tsl::AsyncValueOwningRef<absl::Status>>
    Future<>::ready_promise_(
        tsl::MakeAvailableAsyncValueRef<absl::Status>(ready_promise_storage));

namespace {

class State {
 public:
  State(int32_t size, Promise<> promise)
      : pending_count_(size), promise_(std::move(promise)) {}

  void Update(const absl::Status& status) {
    if (ABSL_PREDICT_FALSE(!status.ok())) {
      absl::MutexLock lock(mu_);
      if (VLOG_IS_ON(2)) {
        if (!status_.ok() && status.code() != status_.code()) {
          VLOG(2) << "Ignoring status " << status << " because first error was "
                  << status_;
        }
      }
      status_.Update(status);
    }

    int32_t pending_count =
        pending_count_.fetch_sub(1, std::memory_order_acq_rel);
    CHECK_GE(pending_count, 1) << "Pending count can't drop below 0";

    if (pending_count == 1) {
      absl::MutexLock lock(mu_);
      promise_.Set(std::move(status_));
    }
  }

 private:
  std::atomic<int32_t> pending_count_;
  Promise<> promise_;

  absl::Mutex mu_;
  absl::Status status_ ABSL_GUARDED_BY(&mu_);
};

}  // namespace

Future<> JoinFutures(absl::Span<const Future<>> futures) {
  VLOG(2) << "tsl::JoinFutures: " << futures.size() << " futures";

  if (futures.empty()) {
    return absl::OkStatus();
  }
  if (futures.size() == 1) {
    return futures.front();
  }

  auto [promise, future] = MakePromise();
  auto state = std::make_shared<State>(futures.size(), std::move(promise));

  for (const Future<>& future : futures) {
    future.OnReady(absl::bind_front(&State::Update, state));
  }

  return std::move(future);
}

}  // namespace tsl
