/* Copyright 2016 The OpenXLA Authors.

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

#include "xla/stream_executor/host/host_platform.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/host/host_executor.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "tsl/platform/status.h"

namespace stream_executor {
namespace host {

HostPlatform::HostPlatform() : name_("Host") {}

HostPlatform::~HostPlatform() {}

Platform::Id HostPlatform::id() const { return kHostPlatformId; }

int HostPlatform::VisibleDeviceCount() const {
  return std::thread::hardware_concurrency();
}

const std::string& HostPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
HostPlatform::DescriptionForDevice(int ordinal) const {
  return HostExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> HostPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  return GetExecutor(config);
}

absl::StatusOr<StreamExecutor*> HostPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
HostPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = std::make_unique<StreamExecutor>(
      this, std::make_unique<HostExecutor>(config.ordinal));
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "failed initializing StreamExecutor for device ordinal %d: %s",
        config.ordinal, init_status.ToString().c_str()));
  }

  return std::move(executor);
}

static void InitializeHostPlatform() {
  std::unique_ptr<Platform> platform(new host::HostPlatform);
  TF_CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace host
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    host_platform, stream_executor::host::InitializeHostPlatform());
