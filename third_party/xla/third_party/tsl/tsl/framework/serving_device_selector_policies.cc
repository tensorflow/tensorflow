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
#include "tsl/framework/serving_device_selector_policies.h"

#include <atomic>

#include "absl/strings/string_view.h"
#include "tsl/framework/serving_device_selector.h"

namespace tsl {

int RoundRobinPolicy::SelectDevice(
    absl::string_view program_fingerprint,
    const ServingDeviceSelector::DeviceStates& device_states) {
  const int num_devices = device_states.states.size();
  return ordinal_.fetch_add(1, std::memory_order_relaxed) % num_devices;
}

}  // namespace tsl
