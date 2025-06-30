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

#ifndef XLA_PJRT_CPU_EXECUTION_STREAM_EVENT_MAP_H_
#define XLA_PJRT_CPU_EXECUTION_STREAM_EVENT_MAP_H_

#include <cstdint>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// A thread-safe map for tracking execution events for execution_stream_id.
class ExecutionStreamEventMap {
 public:
  tsl::AsyncValueRef<CpuEvent> GetLastEnqueueEvent(
      int64_t execution_stream_id) {
    absl::MutexLock lock(&lock_);
    auto iter = map_.find(execution_stream_id);
    if (iter != map_.end()) {
      return iter->second;
    }
    return tsl::MakeAvailableAsyncValueRef<CpuEvent>();
  }

  void SetLastEnqueueEvent(int64_t execution_stream_id,
                           tsl::AsyncValueRef<CpuEvent> event) {
    absl::MutexLock lock(&lock_);
    map_[execution_stream_id] = std::move(event);
  }

  void Clear(int64_t execution_stream_id, tsl::AsyncValuePtr<CpuEvent> event) {
    absl::MutexLock lock(&lock_);

    auto iter = map_.find(execution_stream_id);
    if (iter != map_.end()) {
      if (iter->second.AsPtr() == event) {
        map_.erase(iter);
      }
    }
  }

 private:
  absl::Mutex lock_;
  absl::flat_hash_map<int64_t, tsl::AsyncValueRef<CpuEvent>> map_
      ABSL_GUARDED_BY(lock_);
};

}  // namespace xla

#endif  // XLA_PJRT_CPU_EXECUTION_STREAM_EVENT_MAP_H_
