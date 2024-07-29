/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/framework/serving_device_selector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"

namespace tsl {

inline constexpr int kHighPriority = 0;

DeviceReservation::DeviceReservation(int device_index,
                                     ServingDeviceSelector* device_selector)
    : device_index_(device_index), device_selector_(device_selector) {}

DeviceReservation::~DeviceReservation() { reset(); }

void DeviceReservation::reset() {
  if (device_selector_) device_selector_->FreeDeviceReservation(*this);
  device_selector_ = nullptr;
}

DeviceReservation::DeviceReservation(DeviceReservation&& r)
    : device_index_{r.device_index_}, device_selector_{r.device_selector_} {
  r.device_selector_ = nullptr;
}

DeviceReservation& DeviceReservation::operator=(DeviceReservation&& r) {
  if (this == &r) return *this;

  if (device_selector_) device_selector_->FreeDeviceReservation(*this);

  device_index_ = r.device_index_;
  device_selector_ = r.device_selector_;
  r.device_selector_ = nullptr;
  return *this;
}

/*static*/ void ServingDeviceSelector::CompletedHelper(
    DeviceState& device_state, int32_t device_index, int32_t priority,
    std::optional<int64_t>& min_exec_time, bool had_error, int64_t now_ns) {
  // Check that priority 'priority' queue is non-empty.
  DCHECK(!device_state.enqueued_programs[priority].empty());
  auto& program_info = device_state.enqueued_programs[priority].front();
  auto prefetch_results = program_info.prefetch_results;
  auto execution_info = program_info.execution_info;
  device_state.enqueued_programs[priority].pop_front();
  // To make tracked execution time as accurate as possible, we only record this
  // execution time if two programs ran back-to-back without host round trip.
  if (!device_state.timer_reset && !had_error) {
    VLOG(4) << "Complete. update device[" << device_index
            << "], priority: " << priority
            << ", prefetch: " << static_cast<int>(prefetch_results)
            << ", time: " << now_ns - device_state.last_started_ns;
    const_cast<ExecutionInfo*>(execution_info)
        ->AddTime(now_ns - device_state.last_started_ns, prefetch_results);
    // Only update min_exec_time_ when running_average is updated. This avoids
    // the case where running_average is zero.
    if (!min_exec_time.has_value() ||
        execution_info->GetTime(prefetch_results) < min_exec_time.value()) {
      min_exec_time = execution_info->GetTime(prefetch_results);
    }
  }
  // If there are remaining programs, update the start time.
  if (!device_state.enqueued_programs.empty()) {
    device_state.last_started_ns = now_ns;
    device_state.timer_reset = false;
  }
}

/*static*/ int64_t ServingDeviceSelector::EstimateTimeTillIdleNs(
    const DeviceState& device_state, int32_t priority, int64_t min_exec_time,
    int64_t now_ns) {
  int64_t ns_till_idle = 0;
  // Add time from each program in queues with priority 'priority' or higher.
  for (int32_t i = 0; i <= priority; i++) {
    for (auto& info : device_state.enqueued_programs[i]) {
      ns_till_idle +=
          info.execution_info->MaybeGetValidTime(info.prefetch_results);
    }
  }
  // Accounts for the elapsed time of the currently running but unfinished
  // program (i.e., enqueued programs).
  if (ns_till_idle > 0) {
    DCHECK_GT(device_state.last_started_ns, 0);
    ns_till_idle = std::max<int64_t>(
        0, ns_till_idle - (now_ns - device_state.last_started_ns));
  }

  // Add time from scheduled programs with priority 'priority' or higher
  int64_t ns_of_schedule_programs = 0;
  for (int32_t i = 0; i <= priority; i++) {
    for (auto& info : device_state.scheduled_programs[i]) {
      ns_of_schedule_programs += std::max(
          info.execution_info->MaybeGetValidTime(info.prefetch_results),
          min_exec_time);
    }
  }
  return ns_till_idle + ns_of_schedule_programs;
}
/*static*/ void ServingDeviceSelector::EnqueueHelper(
    DeviceState& device_state, int32_t device_index,
    ExecutionInfo& execution_info, absl::string_view fingerprint,
    int32_t priority, int64_t req_id, size_t priority_queue_count,
    int prefetch_results, int64_t now_ns) {
  if (!device_state.scheduled_programs[priority].empty()) {
    auto& program = device_state.scheduled_programs[priority].front();
    if (program.fingerprint.empty()) {
      program.execution_info = &execution_info;
      program.fingerprint = fingerprint;
      if (priority == kHighPriority) {
        device_state.last_fingerprint = fingerprint;
      }
      device_state.unknown_fingerprint_requests--;
    }
    device_state.enqueued_programs[static_cast<int32_t>(priority)].push_back(
        std::move(program));
    device_state.scheduled_programs[static_cast<int32_t>(priority)].pop_front();
  } else {
    DeviceState::ProgramInfo program;
    program.execution_info = &execution_info;
    program.fingerprint = fingerprint;
    program.req_id = req_id;
    program.priority = priority;
    program.prefetch_results = prefetch_results;
    device_state.enqueued_programs[priority].push_back(program);
    device_state.last_fingerprint = fingerprint;
  }

  // Count number of programs in enqueued_programs queues.
  int64_t num_programs_enqueued = 0;
  for (int64_t i = 0; i < priority_queue_count; i++) {
    num_programs_enqueued += device_state.enqueued_programs[i].size();
  }

  if (num_programs_enqueued == 1) {
    device_state.last_started_ns = now_ns;
    device_state.timer_reset = true;
  }
}
}  // namespace tsl
