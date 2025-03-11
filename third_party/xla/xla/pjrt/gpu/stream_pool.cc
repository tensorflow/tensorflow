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

#include "xla/pjrt/gpu/stream_pool.h"

#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

BoundedStreamPool::Handle::~Handle() {
  if (pool_) {
    pool_->Return(std::move(stream_));
  }
}

BoundedStreamPool::BoundedStreamPool(se::StreamExecutor* executor, int capacity)
    : streams_(capacity) {
  for (int i = 0; i < capacity; ++i) {
    absl::StatusOr<std::unique_ptr<se::Stream>> stream =
        executor->CreateStream();
    CHECK_OK(stream) << "Failed to create stream in bounded stream pool";
    streams_.push_back(std::move(*stream));
  }
}

absl::StatusOr<BoundedStreamPool::Handle> BoundedStreamPool::Borrow() {
  absl::MutexLock lock(&mu_);
  auto stream_available = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !streams_.empty();
  };
  mu_.Await(absl::Condition(&stream_available));
  std::unique_ptr<se::Stream> stream = std::move(streams_.back());
  streams_.pop_back();
  return Handle(this, std::move(stream));
}

void BoundedStreamPool::Return(std::unique_ptr<se::Stream> stream) {
  absl::MutexLock lock(&mu_);
  // TODO(phawkins): consider verifying streams are error free before returning
  // them to the pool.
  streams_.push_back(std::move(stream));
}

}  // namespace xla
