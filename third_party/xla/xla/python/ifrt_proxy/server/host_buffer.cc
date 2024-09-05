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

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace xla {
namespace ifrt {
namespace proxy {

absl::Status HostBufferStore::Store(uint64_t handle, std::string data) {
  absl::MutexLock lock(&mu_);
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
    uint64_t handle) {
  absl::MutexLock lock(&mu_);
  const auto it = buffers_.find(handle);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Host buffer handle ", handle, " not found"));
  }
  return it->second;
}

absl::Status HostBufferStore::Delete(uint64_t handle) {
  absl::MutexLock lock(&mu_);
  if (buffers_.erase(handle) == 0) {
    return absl::NotFoundError(
        absl::StrCat("Host buffer handle ", handle, " not found"));
  }
  return absl::OkStatus();
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
