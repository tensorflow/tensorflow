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
#include "tensorflow/core/common_runtime/gpu/gpu_serving_device_selector.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {
namespace gpu {

ABSL_CONST_INIT int64_t (*NowNs)() = +[]() -> int64_t {
  return absl::GetCurrentTimeNanos();
};

using DeviceStates = GpuServingDeviceSelector::DeviceStates;

GpuServingDeviceSelector::GpuServingDeviceSelector(
    const int num_devices,
    std::unique_ptr<ServingDeviceSelector::Policy> device_selector_policy)
    : device_states_(num_devices),
      device_selector_policy_(std::move(device_selector_policy)),
      req_id_counter_(0) {}

DeviceReservation GpuServingDeviceSelector::ReserveDevice(
    absl::string_view program_fingerprint) {
  absl::MutexLock lock(&mu_);
  DeviceStates device_states;
  device_states.states = absl::Span<const DeviceState>(device_states_);
  const int device_index =
      device_selector_policy_->SelectDevice(program_fingerprint, device_states);

  DeviceState::ProgramInfo program_info;
  program_info.fingerprint = program_fingerprint;
  program_info.req_id = ++req_id_counter_;
  device_states_[device_index].enqueued_programs[0].push_back(program_info);

  return DeviceReservation(device_index, this);
}

void GpuServingDeviceSelector::FreeDeviceReservation(
    const DeviceReservation& reservation) {
  Completed(reservation.device_index());
}

void GpuServingDeviceSelector::Enqueue(int32_t index_on_host,
                                       absl::string_view fingerprint) {
  if (fingerprint.empty()) {
    LOG(ERROR) << "Empty fingerprint.";
    return;
  }

  absl::MutexLock lock(&mu_);
  auto [it, emplaced] =
      execution_info_.try_emplace(fingerprint, ExecutionInfo());

  DeviceState& device_state = device_states_.at(index_on_host);
  DeviceState::ProgramInfo program_info;
  program_info.fingerprint = fingerprint;
  program_info.execution_info = &(it->second);
  device_state.enqueued_programs[0].push_back(program_info);

  auto num_programs = device_state.enqueued_programs[0].size();
  if (num_programs == 1) {
    device_state.last_started_ns = NowNs();
    device_state.timer_reset = true;
  }

  // TODO(xiangll): Metric estimated execution time.
}

void GpuServingDeviceSelector::Completed(int32_t index_on_host,
                                         bool had_error) {
  absl::MutexLock lock(&mu_);
  DeviceState& device_state = device_states_.at(index_on_host);

  DCHECK(!device_state.enqueued_programs[0].empty());
  auto& program_info = device_state.enqueued_programs[0].front();
  auto& execution_info = program_info.execution_info;
  device_state.enqueued_programs[0].pop_front();
  const int64_t now_ns = NowNs();
  // To make tracked execution time as accurate as possible, we only record this
  // execution time if two programs ran back-to-back without host round trip.
  if (!device_state.timer_reset && !had_error) {
    const_cast<ExecutionInfo*>(execution_info)
        ->AddTime(now_ns - device_state.last_started_ns);
  }
  // If there are remaining programs, update the start time.
  if (!device_state.enqueued_programs[0].empty()) {
    device_state.last_started_ns = now_ns;
    device_state.timer_reset = false;
  }

  // TODO(xiangll): Metric estimated execution time.
}

/*static*/ int64_t GpuServingDeviceSelector::EstimateTimeTillIdleNs(
    const DeviceState& device_state) {
  int64_t ns_till_idle = 0;
  // Add time from each program in queue.
  for (const auto& program_info : device_state.enqueued_programs[0]) {
    ns_till_idle += program_info.execution_info->GetTime();
  }

  // Accounts for the elapsed time of the currently running but unfinished
  // program.
  if (ns_till_idle > 0) {
    DCHECK_GT(device_state.last_started_ns, 0);
    ns_till_idle = std::max<int64_t>(
        0, ns_till_idle - (NowNs() - device_state.last_started_ns));
  }

  return ns_till_idle;
}

int64_t GpuServingDeviceSelector::TotalGpuLoadNsForTest() {
  absl::MutexLock lock(&mu_);
  int64_t total_gpu_load_ns = 0;
  for (const auto& device_state : device_states_) {
    total_gpu_load_ns += EstimateTimeTillIdleNs(device_state);
  }
  return total_gpu_load_ns;
}

/*static*/ void GpuServingDeviceSelector::OverwriteNowNsFunctionForTest(
    int64_t (*now_ns)()) {
  NowNs = now_ns;
}
}  // namespace gpu
}  // namespace tensorflow
