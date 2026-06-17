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

#include "xla/pjrt/profiling/no_op_device_time_measurement.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/time/time.h"
#include "xla/pjrt/profiling/device_time_measurement.h"

namespace xla {

std::unique_ptr<DeviceTimeMeasurement> CreateDeviceTimeMeasurement() {
  return std::make_unique<NoOpDeviceTimeMeasurement>();
}

std::optional<uint64_t> GetDeviceTimeMeasurementKey() { return std::nullopt; }

void RecordDeviceTimeMeasurement(
    uint64_t key, absl::Duration elapsed,
    xla::DeviceTimeMeasurement::DeviceType device_type) {}

}  // namespace xla
