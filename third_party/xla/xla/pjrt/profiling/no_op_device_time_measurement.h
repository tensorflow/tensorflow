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

#ifndef XLA_PJRT_PROFILING_NO_OP_DEVICE_TIME_MEASUREMENT_H_
#define XLA_PJRT_PROFILING_NO_OP_DEVICE_TIME_MEASUREMENT_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "xla/pjrt/profiling/device_time_measurement.h"

namespace xla {

// No-op OSS implementation of DeviceTimeMeasurement.
class NoOpDeviceTimeMeasurement : public DeviceTimeMeasurement {
 public:
  NoOpDeviceTimeMeasurement() = default;
  ~NoOpDeviceTimeMeasurement() override = default;

  NoOpDeviceTimeMeasurement(const NoOpDeviceTimeMeasurement&) = delete;
  NoOpDeviceTimeMeasurement& operator=(const NoOpDeviceTimeMeasurement&) =
      delete;

  // Get the total device duration of the input device type (either GPU or TPU)
  // since the creation of the DeviceTimeMeasurement object.
  absl::Duration GetTotalDuration(DeviceType device_type) override {
    return absl::ZeroDuration();
  };

  // Get the total device durations of all device types (GPU and TPU)
  // since the creation of the DeviceTimeMeasurement object.
  absl::flat_hash_map<DeviceType, absl::Duration> GetTotalDurations() override {
    return device_type_durations_;
  }

  // Record elapsed device time for the given input device type.
  void Record(absl::Duration elapsed, DeviceType device_type) override {};
};

inline std::unique_ptr<DeviceTimeMeasurement> CreateDeviceTimeMeasurement() {
  return std::make_unique<NoOpDeviceTimeMeasurement>();
}

inline std::optional<uint64_t> GetDeviceTimeMeasurementKey() {
  return std::nullopt;
}

inline void RecordDeviceTimeMeasurement(
    uint64_t key, absl::Duration elapsed,
    xla::DeviceTimeMeasurement::DeviceType device_type) {}

}  // namespace xla
#endif  // XLA_PJRT_PROFILING_NO_OP_DEVICE_TIME_MEASUREMENT_H_
