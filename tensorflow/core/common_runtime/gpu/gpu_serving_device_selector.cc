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

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {
namespace gpu {

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
  device_states_[device_index].scheduled_programs.push_back(program_info);

  return DeviceReservation(device_index, this);
}

void GpuServingDeviceSelector::FreeDeviceReservation(
    const DeviceReservation& reservation) {
  absl::MutexLock lock(&mu_);
  auto& scheduled_programs =
      device_states_.at(reservation.device_index()).scheduled_programs;
  DCHECK(!scheduled_programs.empty());
  scheduled_programs.pop_front();
}

}  // namespace gpu
}  // namespace tensorflow
