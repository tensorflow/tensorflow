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

#ifndef XLA_PJRT_PROFILING_TEST_UTIL_MOCK_DEVICE_TIME_MEASUREMENT_H_
#define XLA_PJRT_PROFILING_TEST_UTIL_MOCK_DEVICE_TIME_MEASUREMENT_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "xla/pjrt/profiling/device_time_measurement.h"

namespace xla {

// Mock implementation of DeviceTimeMeasurement for testing.
//
// MockDeviceTimeMeasurement objects are tracked in a global stack. This should
// only be used by local variables so that objects are destructed in the reverse
// order of their creation.
class MockDeviceTimeMeasurement : public DeviceTimeMeasurement {
 public:
  // Adds a MockDeviceTimeMeasurement object to the top of the global stack.
  MockDeviceTimeMeasurement();

  // Removes the top-most MockDeviceTimeMeasurement object from
  // the global stack.
  ~MockDeviceTimeMeasurement() override;

  MockDeviceTimeMeasurement(const MockDeviceTimeMeasurement&) = delete;
  MockDeviceTimeMeasurement& operator=(const MockDeviceTimeMeasurement&) =
      delete;

  // Get the total device duration of the input device type (either GPU or TPU)
  // since the creation of the MockDeviceTimeMeasurement object.
  absl::Duration GetTotalDuration(DeviceType device_type) override {
    return absl::ZeroDuration();
  };

  // Get the total device durations of all device types (GPU and TPU)
  // since the creation of the MockDeviceTimeMeasurement object.
  absl::flat_hash_map<DeviceType, absl::Duration> GetTotalDurations() override {
    return device_type_durations_;
  }

  // Record elapsed device time for the given input device type.
  void Record(absl::Duration elapsed, DeviceType device_type) override {};
};

std::unique_ptr<DeviceTimeMeasurement> CreateDeviceTimeMeasurement();

// Returns the key (position) of the top-most MockDeviceTimeMeasurement object
// in the global stack.
std::optional<uint64_t> GetDeviceTimeMeasurementKey();

// Records the elapsed device time for the MockDeviceTimeMeasurement at the
// given key (position) in the global stack.
void RecordDeviceTimeMeasurement(
    uint64_t key, absl::Duration elapsed,
    xla::DeviceTimeMeasurement::DeviceType device_type);

}  // namespace xla
#endif  // XLA_PJRT_PROFILING_TEST_UTIL_MOCK_DEVICE_TIME_MEASUREMENT_H_
