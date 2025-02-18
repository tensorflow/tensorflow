/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/execution_tracker.h"

#include <memory>
#include <utility>

#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {

AsyncExecution::AsyncExecution(Backend* backend,
                               std::vector<StreamPool::Ptr> streams,
                               const ExecutionProfile& profile,
                               GlobalDataHandle result)
    : backend_(CHECK_NOTNULL(backend)),
      streams_(std::move(streams)),
      profile_(profile),
      result_(std::move(result)) {
  for (const auto& stream : streams_) {
    CHECK(stream != nullptr);
  }
}

absl::Status AsyncExecution::BlockUntilDone() const {
  for (auto& stream : streams_) {
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  }
  return absl::OkStatus();
}

ExecutionTracker::ExecutionTracker() : next_handle_(1) {}

ExecutionHandle ExecutionTracker::Register(Backend* backend,
                                           std::vector<StreamPool::Ptr> streams,
                                           const ExecutionProfile& profile,
                                           GlobalDataHandle result) {
  absl::MutexLock lock(&execution_mutex_);
  int64_t handle = next_handle_++;
  auto inserted = handle_to_execution_.emplace(
      handle, std::make_unique<AsyncExecution>(backend, std::move(streams),
                                               profile, result));
  CHECK(inserted.second);

  ExecutionHandle execution_handle;
  execution_handle.set_handle(handle);
  return execution_handle;
}

absl::Status ExecutionTracker::Unregister(const ExecutionHandle& handle) {
  absl::MutexLock lock(&execution_mutex_);
  auto it = handle_to_execution_.find(handle.handle());
  if (it == handle_to_execution_.end()) {
    return NotFound("no execution record for execution handle: %d",
                    handle.handle());
  }
  handle_to_execution_.erase(handle.handle());
  return absl::OkStatus();
}

absl::StatusOr<const AsyncExecution*> ExecutionTracker::Resolve(
    const ExecutionHandle& handle) {
  absl::MutexLock lock(&execution_mutex_);
  auto it = handle_to_execution_.find(handle.handle());
  if (it == handle_to_execution_.end()) {
    return NotFound("no execution record for execution handle: %d",
                    handle.handle());
  }
  return it->second.get();
}

}  // namespace xla
