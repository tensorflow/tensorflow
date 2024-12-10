// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/server/host_buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

absl::Status HostBufferStore::Store(uint64_t handle, std::string data) {
  VLOG(3) << "HostBuffer::Store " << handle << " " << data.size();
  absl::MutexLock lock(&mu_);
  if (shutdown_msg_.has_value()) {
    return absl::CancelledError(*shutdown_msg_);
  }
  const bool inserted =
      buffers_.insert({handle, std::make_shared<std::string>(std::move(data))})
          .second;
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Host buffer handle ", handle, " already exists"));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<const std::string>> HostBufferStore::Lookup(
    uint64_t handle, absl::Duration timeout) {
  VLOG(3) << "HostBufferStore::Lookup " << handle
          << " start, timeout=" << timeout;
  tsl::profiler::TraceMe traceme("HostBufferStore::Lookup");
  auto result = [&]() -> absl::StatusOr<std::shared_ptr<const std::string>> {
    absl::MutexLock lock(&mu_);
    auto cond = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return shutdown_msg_.has_value() || buffers_.contains(handle);
    };
    if (!cond()) {
      tsl::profiler::TraceMe traceme("HostBufferStore::Lookup.Wait");
      mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
    }
    if (shutdown_msg_) {
      return absl::CancelledError(shutdown_msg_.value());
    }
    const auto it = buffers_.find(handle);
    if (it == buffers_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Host buffer handle ", handle, " not found"));
    }
    return it->second;
  }();
  if (result.ok()) {
    VLOG(3) << "HostBufferStore::Lookup " << handle
            << " done, size=" << (*result == nullptr ? -1 : (*result)->size());
  } else {
    VLOG(3) << "HostBufferStore::Lookup " << handle
            << " done: " << result.status();
  }
  return result;
}

absl::Status HostBufferStore::Delete(uint64_t handle) {
  absl::MutexLock lock(&mu_);
  VLOG(3) << "HostBufferStore::Delete " << handle;
  if (buffers_.erase(handle) == 0) {
    return absl::NotFoundError(
        absl::StrCat("Host buffer handle ", handle, " not found"));
  }
  return absl::OkStatus();
}

void HostBufferStore::Shutdown(std::string reason) {
  VLOG(0) << "HostBufferStore::Shutdown " << reason;
  absl::MutexLock lock(&mu_);
  if (!shutdown_msg_.has_value()) {
    shutdown_msg_ = std::move(reason);
  }
  buffers_.clear();
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
