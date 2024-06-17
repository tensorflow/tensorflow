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
#include "xla/tsl/framework/serving_device_selector.h"
#include "tensorflow/core/common_runtime/gpu/gpu_scheduling_metrics_storage.h"

namespace tensorflow {
namespace gpu {
// A default estimate of execution time for an enqueued program that this host
// has never finished executing. We currently set it to 1 ns (so that for all
// empty queues it still affects the decision) until we have better way to
// estimate this, as this penalty is chip-dependent and program-dependent.
constexpr int64_t kDefaultEstimateNs = 1;
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

tsl::DeviceReservation GpuServingDeviceSelector::ReserveDevice(
    absl::string_view program_fingerprint) {
  absl::MutexLock lock(&mu_);
  DeviceStates device_states;
  device_states.states = absl::Span<const DeviceState>(device_states_);
  auto [it, emplaced] =
      execution_info_.try_emplace(program_fingerprint, ExecutionInfo());
  const int device_index =
      device_selector_policy_->SelectDevice(program_fingerprint, device_states);

  ServingDeviceSelector::EnqueueHelper(
      device_states_.at(device_index), device_index, it->second,
      program_fingerprint, /*priority=*/0, req_id_counter_++,
      /*priority_queue_count=*/1, /*prefetch_results=*/0, NowNs());

  return tsl::DeviceReservation(device_index, this);
}

void GpuServingDeviceSelector::FreeDeviceReservation(
    const tsl::DeviceReservation& reservation) {
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
  ServingDeviceSelector::EnqueueHelper(device_state, index_on_host, it->second,
                                       fingerprint,
                                       /*priority=*/0, /*req_id=*/-1,
                                       /*priority_queue_count=*/1,
                                       /*prefetch_results=*/0, NowNs());

  int64_t total_estimated_time_ns = TotalEstimatedTimeTillIdleNs();
  GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Set(
      total_estimated_time_ns);
}

void GpuServingDeviceSelector::Completed(int32_t index_on_host,
                                         bool had_error) {
  absl::MutexLock lock(&mu_);
  DeviceState& device_state = device_states_.at(index_on_host);
  ServingDeviceSelector::CompletedHelper(device_state, index_on_host, 0,
                                         min_exec_time_, had_error, NowNs());

  int64_t total_estimated_time_ns = TotalEstimatedTimeTillIdleNs();
  GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Set(
      total_estimated_time_ns);
}

int64_t GpuServingDeviceSelector::TotalEstimatedTimeTillIdleNs() {
  int64_t total_gpu_load_ns = 0;
  for (const auto& device_state : device_states_) {
    total_gpu_load_ns += ServingDeviceSelector::EstimateTimeTillIdleNs(
        device_state, 0, min_exec_time_.value_or(kDefaultEstimateNs), NowNs());
  }
  return total_gpu_load_ns;
}

/*static*/ void GpuServingDeviceSelector::OverwriteNowNsFunctionForTest(
    int64_t (*now_ns)()) {
  NowNs = now_ns;
}

}  // namespace gpu
}  // namespace tensorflow
