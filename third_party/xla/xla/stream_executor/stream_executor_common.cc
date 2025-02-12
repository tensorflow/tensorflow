/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/stream_executor_common.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/status.h"

namespace stream_executor {

// Get per-device memory limit in bytes. Returns 0 if
// TF_PER_DEVICE_MEMORY_LIMIT_MB environment variable is not set.
static int64_t GetMemoryLimitBytesFromEnvironmentVariable() {
  int64_t value;
  TF_CHECK_OK(
      tsl::ReadInt64FromEnvVar("TF_PER_DEVICE_MEMORY_LIMIT_MB", 0, &value));
  return value * (1ll << 20);
}

StreamExecutorCommon::StreamExecutorCommon(const Platform* platform)
    : platform_(platform),
      memory_limit_bytes_(GetMemoryLimitBytesFromEnvironmentVariable()) {}

const DeviceDescription& StreamExecutorCommon::GetDeviceDescription() const {
  absl::MutexLock lock(&mu_);
  if (device_description_ != nullptr) {
    return *device_description_;
  }

  device_description_ = CreateDeviceDescription().value();
  return *device_description_;
}

}  // namespace stream_executor
