/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/host/host_platform.h"

#include <thread>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/platform/initialize.h"
#include "tensorflow/tsl/platform/errors.h"

namespace stream_executor {
namespace host {

HostPlatform::HostPlatform() : name_("Host") {}

HostPlatform::~HostPlatform() {}

Platform::Id HostPlatform::id() const { return kHostPlatformId; }

int HostPlatform::VisibleDeviceCount() const {
  return std::thread::hardware_concurrency();
}

const std::string& HostPlatform::Name() const { return name_; }

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
HostPlatform::DescriptionForDevice(int ordinal) const {
  return HostExecutor::CreateDeviceDescription(ordinal);
}

tsl::StatusOr<StreamExecutor*> HostPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

tsl::StatusOr<StreamExecutor*> HostPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

tsl::StatusOr<StreamExecutor*> HostPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

tsl::StatusOr<std::unique_ptr<StreamExecutor>>
HostPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = std::make_unique<StreamExecutor>(
      this, std::make_unique<HostExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return tsl::Status(
        tsl::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str()));
  }

  return std::move(executor);
}

void HostPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register host trace listener";
}

void HostPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister host trace listener";
}

static void InitializeHostPlatform() {
  std::unique_ptr<Platform> platform(new host::HostPlatform);
  TF_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace host
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(host_platform,
                            stream_executor::host::InitializeHostPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(host_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     host_platform);
