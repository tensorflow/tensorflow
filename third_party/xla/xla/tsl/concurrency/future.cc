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

#include <memory>
#include <utility>
#include <variant>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
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

// A state for tracking `JoinFutures` for stateless `Future<>`.
class JoinStateless : public internal::JoinFutures<JoinStateless> {
 public:
  using internal::JoinFutures<JoinStateless>::JoinFutures;

  void OnReady(const absl::Status& status) {
    Update(status, [](std::monostate) { /* no state to update */ });
  }

  void Complete(Promise<> promise, absl::Status status, std::monostate) {
    promise.Set(std::move(status));
  }
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
  auto join =
      std::make_shared<JoinStateless>(futures.size(), std::move(promise));

  for (const Future<>& future : futures) {
    future.OnReady(
        [join](const absl::Status& status) { join->OnReady(status); });
  }

  return std::move(future);
}

}  // namespace tsl
