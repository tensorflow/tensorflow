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
#include "xla/pjrt/profiling/test_util/mock_device_time_measurement.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/debugging/leak_check.h"
#include "absl/time/time.h"
#include "xla/pjrt/profiling/device_time_measurement.h"

// Global vector for tracking MockDeviceTimeMeasurement objects.
std::vector<xla::MockDeviceTimeMeasurement*>* mock_device_time_measurements() {
  // Declaring a global absl::NoDestructor<Overrides> is easier, but as of Feb
  // 2025, NoDestructor<> was not yet available in the version of absl linked
  // into TSL.
  static std::vector<xla::MockDeviceTimeMeasurement*>*
      mock_device_time_measurements =
          new std::vector<xla::MockDeviceTimeMeasurement*>();
  absl::IgnoreLeak(mock_device_time_measurements);
  return mock_device_time_measurements;
}

namespace xla {

MockDeviceTimeMeasurement::MockDeviceTimeMeasurement() {
  mock_device_time_measurements()->push_back(this);
}

MockDeviceTimeMeasurement::~MockDeviceTimeMeasurement() {
  mock_device_time_measurements()->pop_back();
}

std::unique_ptr<DeviceTimeMeasurement> CreateDeviceTimeMeasurement() {
  return std::make_unique<MockDeviceTimeMeasurement>();
}

std::optional<uint64_t> GetDeviceTimeMeasurementKey() {
  return mock_device_time_measurements()->size() - 1;
}

void RecordDeviceTimeMeasurement(
    uint64_t key, absl::Duration elapsed,
    xla::DeviceTimeMeasurement::DeviceType device_type) {
  (*mock_device_time_measurements())[key]->Record(elapsed, device_type);
}

}  // namespace xla
