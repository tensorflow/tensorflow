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

#ifndef XLA_PJRT_PROFILING_DEVICE_TIME_MEASUREMENT_H_
#define XLA_PJRT_PROFILING_DEVICE_TIME_MEASUREMENT_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_compiler.h"

namespace xla {

// Interface for measuring accelerator (TPU and GPU) device time.
class DeviceTimeMeasurement {
 public:
  enum class DeviceType {
    kUnknown,
    kTpu,
    kGpu,
  };

  DeviceTimeMeasurement() = default;
  virtual ~DeviceTimeMeasurement() = default;

  DeviceTimeMeasurement(const DeviceTimeMeasurement&) = delete;
  DeviceTimeMeasurement& operator=(const DeviceTimeMeasurement&) = delete;

  // Get the total device duration of the input device type (either GPU or TPU)
  // since the creation of the DeviceTimeMeasurement object.
  virtual absl::Duration GetTotalDuration(DeviceType device_type) = 0;

  // Get the total device durations of all device types (GPU and TPU)
  // since the creation of the DeviceTimeMeasurement object.
  virtual absl::flat_hash_map<DeviceType, absl::Duration>
  GetTotalDurations() = 0;

  // Record elapsed device time for the given input device type.
  virtual void Record(absl::Duration elapsed, DeviceType device_type) = 0;

 protected:
  absl::Mutex mu_;
  absl::flat_hash_map<DeviceType, absl::Duration> device_type_durations_ = {
      {DeviceType::kTpu, absl::ZeroDuration()},
      {DeviceType::kGpu, absl::ZeroDuration()},
  };
};

// Factory function for creating a DeviceTimeMeasurement object.
std::unique_ptr<DeviceTimeMeasurement> CreateDeviceTimeMeasurement();

// Helper function to retrieve current DeviceTimeMeasurement's opaque
// key from Context. The key can be passed when reporting measurements via
// `RecordDeviceTimeMeasurement`.
//
// Returns std::nullopt for OSS and cloud builds.
std::optional<uint64_t> GetDeviceTimeMeasurementKey();

// Records device measurement of the given device_type to the
// `DeviceTimeMeasurement` associated with the given key from
// `GetDeviceTimeMeasurementKey`. Does nothing if the
// `DeviceTimeMeasurement` associated with the key does not exist.
void RecordDeviceTimeMeasurement(
    uint64_t key, absl::Duration elapsed,
    xla::DeviceTimeMeasurement::DeviceType device_type);

// Helper function to convert PjRtPlatformId to
// DeviceTimeMeasurement::DeviceType.
inline DeviceTimeMeasurement::DeviceType GetDeviceType(
    PjRtPlatformId platform_id) {
  if (platform_id == CudaId() || platform_id == RocmId() ||
      platform_id == SyclId()) {
    return DeviceTimeMeasurement::DeviceType::kGpu;
  } else if (platform_id == TpuId()) {
    return DeviceTimeMeasurement::DeviceType::kTpu;
  }
  return DeviceTimeMeasurement::DeviceType::kUnknown;
}

}  // namespace xla
#endif  // XLA_PJRT_PROFILING_DEVICE_TIME_MEASUREMENT_H_
