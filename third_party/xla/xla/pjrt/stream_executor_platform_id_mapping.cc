/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/stream_executor_platform_id_mapping.h"

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {

StreamExecutorPlatformIdMapping& StreamExecutorPlatformIdMapping::Global() {
  static absl::NoDestructor<StreamExecutorPlatformIdMapping> instance;
  return *instance;
}
absl::StatusOr<PjRtPlatformId>
StreamExecutorPlatformIdMapping::GetPjRtPlatformId(
    stream_executor::PlatformId stream_executor_platform_id) const {
  absl::MutexLock lock(mutex_);
  if (auto it = stream_executor_to_pjrt_platform_id_.find(
          stream_executor_platform_id);
      it != stream_executor_to_pjrt_platform_id_.end()) {
    return it->second;
  }
  return absl::NotFoundError(absl::StrCat("StreamExecutor platform ID ",
                                          stream_executor_platform_id->ToName(),
                                          " not found in mapping."));
}
absl::StatusOr<stream_executor::PlatformId>
StreamExecutorPlatformIdMapping::GetStreamExecutorPlatformId(
    PjRtPlatformId pjrt_platform_id) const {
  absl::MutexLock lock(mutex_);
  if (auto it = pjrt_to_stream_executor_platform_id_.find(pjrt_platform_id);
      it != pjrt_to_stream_executor_platform_id_.end()) {
    return it->second;
  }
  return absl::NotFoundError(absl::StrCat("PjRt platform ID ", pjrt_platform_id,
                                          " not found in mapping."));
}

absl::Status StreamExecutorPlatformIdMapping::AddMapping(
    stream_executor::PlatformId stream_executor_platform_id,
    PjRtPlatformId pjrt_platform_id) {
  absl::MutexLock lock(mutex_);
  if (stream_executor_to_pjrt_platform_id_.contains(
          stream_executor_platform_id)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "StreamExecutor platform ID ", stream_executor_platform_id->ToName(),
        " already exists in mapping."));
  }
  if (pjrt_to_stream_executor_platform_id_.contains(pjrt_platform_id)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "PjRt platform ID ", pjrt_platform_id, " already exists in mapping."));
  }
  stream_executor_to_pjrt_platform_id_[stream_executor_platform_id] =
      pjrt_platform_id;
  pjrt_to_stream_executor_platform_id_[pjrt_platform_id] =
      stream_executor_platform_id;
  return absl::OkStatus();
}

}  // namespace xla
