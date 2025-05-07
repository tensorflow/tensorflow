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

#include "xla/pjrt/gpu/tfrt/stream_pool.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/profiler/lib/traceme.h"

namespace se = ::stream_executor;

namespace xla {

BoundedStreamPool::BoundedStreamPool(se::StreamExecutor* executor,
                                     int capacity) {
  CHECK_GT(capacity, 0) << "Capacity must be positive";
  for (int i = 0; i < capacity; ++i) {
    absl::StatusOr<std::unique_ptr<se::Stream>> stream =
        executor->CreateStream();
    CHECK_OK(stream) << "Failed to create stream in bounded stream pool";
    CHECK_NE(stream->get(), nullptr)
        << "Created a null stream in bounded stream pool";
    streams_.push_back(*std::move(stream));
  }
}

absl::StatusOr<BoundedStreamPool::Handle> BoundedStreamPool::Borrow() {
  tsl::profiler::TraceMe t("BoundedStreamPool::Borrow");

  absl::MutexLock lock(&mu_);
  auto stream_available = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !streams_.empty();
  };
  mu_.Await(absl::Condition(&stream_available));
  std::unique_ptr<se::Stream> stream = std::move(streams_.back());
  streams_.pop_back();
  CHECK_NE(stream.get(), nullptr)
      << "Borrowed a null stream from the pool, which is not allowed.";
  return Handle(this, std::move(stream));
}

size_t BoundedStreamPool::GetAvailableStreamNum()
    ABSL_SHARED_LOCKS_REQUIRED(&mu_) {
  return streams_.size();
}

void BoundedStreamPool::Return(std::unique_ptr<se::Stream> stream) {
  absl::MutexLock lock(&mu_);
  if (stream->ok()) {
    streams_.push_back(std::move(stream));
  }
}

}  // namespace xla
