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

#include "xla/pjrt/cpu/cpu_async_execution_tracker.h"

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

CpuScopedAsyncExecution::CpuScopedAsyncExecution(
    CpuAsyncExecutionTracker* tracker, int32_t launch_id, Key key)
    : tracker_(tracker), launch_id_(launch_id), key_(key) {}

CpuScopedAsyncExecution::CpuScopedAsyncExecution(
    CpuScopedAsyncExecution&& other)
    : tracker_(other.tracker_), launch_id_(other.launch_id_), key_(other.key_) {
  other.tracker_ = nullptr;
}

CpuScopedAsyncExecution::~CpuScopedAsyncExecution() {
  if (tracker_ != nullptr) {
    tracker_->RemoveAsyncExecution(launch_id_, key_);
  }
}

void CpuScopedAsyncExecution::SetStateConcrete() {
  if (tracker_ != nullptr) {
    tracker_->SetStateConcrete(launch_id_, key_);
    tracker_ = nullptr;
  }
}

void CpuScopedAsyncExecution::SetError(absl::Status error) {
  if (tracker_ != nullptr) {
    tracker_->SetError(launch_id_, key_, std::move(error));
    tracker_ = nullptr;
  }
}

CpuScopedAsyncExecution CpuAsyncExecutionTracker::NewAsyncExecution(
    int32_t launch_id, tsl::AsyncValueRef<CpuEvent> execute_event) {
  absl::MutexLock lock(&mu_);
  Key async_execution_key = execute_event.GetAsyncValue();
  executions_[launch_id].insert(
      {async_execution_key, std::move(execute_event)});
  return CpuScopedAsyncExecution(this, launch_id, async_execution_key);
}

bool CpuAsyncExecutionTracker::SetError(int32_t launch_id, absl::Status error) {
  absl::ReleasableMutexLock lock(&mu_);
  auto it = executions_.find(launch_id);
  if (it != executions_.end()) {
    absl::flat_hash_map<Key, tsl::AsyncValueRef<CpuEvent>> execute_events =
        std::move(it->second);
    executions_.erase(it);
    lock.Release();

    if (execute_events.size() == 1) {
      // Fast path for an execution with a unique `launch_id`.
      tsl::AsyncValueRef<CpuEvent>& execute_event =
          execute_events.begin()->second;
      if (execute_event.IsUnavailable()) {
        execute_event.SetError(std::move(error));
        return true;
      }
      return false;
    } else {
      bool any_success = false;
      for (auto& [key, execute_event] : execute_events) {
        if (execute_event.IsUnavailable()) {
          execute_event.SetError(error);
          any_success = true;
        }
      }
      return any_success;
    }
  }
  return false;
}

void CpuAsyncExecutionTracker::SetError(int32_t launch_id, Key key,
                                        absl::Status error) {
  absl::ReleasableMutexLock lock(&mu_);
  auto it = executions_.find(launch_id);
  if (it != executions_.end()) {
    auto it2 = it->second.find(key);
    if (it2 != it->second.end()) {
      tsl::AsyncValueRef<CpuEvent> execute_event = std::move(it2->second);
      it->second.erase(it2);
      if (it->second.empty()) {
        executions_.erase(it);
      }
      lock.Release();

      if (execute_event.IsUnavailable()) {
        execute_event.SetError(error);
      }
    }
  }
}

void CpuAsyncExecutionTracker::SetStateConcrete(int32_t launch_id, Key key) {
  absl::ReleasableMutexLock lock(&mu_);
  auto it = executions_.find(launch_id);
  if (it != executions_.end()) {
    auto it2 = it->second.find(key);
    if (it2 != it->second.end()) {
      tsl::AsyncValueRef<CpuEvent> execute_event = std::move(it2->second);
      it->second.erase(it2);
      if (it->second.empty()) {
        executions_.erase(it);
      }
      lock.Release();

      if (execute_event.IsUnavailable()) {
        execute_event.SetStateConcrete();
      }
    }
  }
}

void CpuAsyncExecutionTracker::RemoveAsyncExecution(int32_t launch_id,
                                                    Key key) {
  absl::MutexLock lock(&mu_);
  auto it = executions_.find(launch_id);
  if (it != executions_.end()) {
    auto it2 = it->second.find(key);
    if (it2 != it->second.end()) {
      it->second.erase(it2);
      if (it->second.empty()) {
        executions_.erase(it);
      }
    }
  }
}

}  // namespace xla
