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

#ifndef XLA_PJRT_STREAM_EXECUTOR_PLATFORM_ID_MAPPING_H_
#define XLA_PJRT_STREAM_EXECUTOR_PLATFORM_ID_MAPPING_H_

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {

/** A mapping between StreamExecutor platform IDs and PjRt platform IDs.
 *
 * Allows to look up the PjRt platform ID given a StreamExecutor platform ID and
 * vice versa. This is useful for libraries that are registered with both
 * StreamExecutor and PjRt and need to interoperate with each other.
 */
class StreamExecutorPlatformIdMapping {
 public:
  static StreamExecutorPlatformIdMapping& Global();

  absl::StatusOr<PjRtPlatformId> GetPjRtPlatformId(
      stream_executor::PlatformId stream_executor_platform_id) const;

  absl::StatusOr<stream_executor::PlatformId> GetStreamExecutorPlatformId(
      PjRtPlatformId pjrt_platform_id) const;

  absl::Status AddMapping(
      stream_executor::PlatformId stream_executor_platform_id,
      PjRtPlatformId pjrt_platform_id);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<stream_executor::PlatformId, PjRtPlatformId>
      stream_executor_to_pjrt_platform_id_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<PjRtPlatformId, stream_executor::PlatformId>
      pjrt_to_stream_executor_platform_id_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_PJRT_STREAM_EXECUTOR_PLATFORM_ID_MAPPING_H_
