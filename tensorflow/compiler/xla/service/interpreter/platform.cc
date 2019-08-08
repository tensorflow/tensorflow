/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/interpreter/platform.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/interpreter/executor.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {
namespace interpreter {

XlaInterpreterPlatform::XlaInterpreterPlatform(const string& name,
                                               const Platform::Id& id)
    : name_(name), id_(id) {}

XlaInterpreterPlatform::~XlaInterpreterPlatform() {}

Platform::Id XlaInterpreterPlatform::id() const { return id_; }

int XlaInterpreterPlatform::VisibleDeviceCount() const { return 1; }

const string& XlaInterpreterPlatform::Name() const { return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
XlaInterpreterPlatform::DescriptionForDevice(int ordinal) const {
  return XlaInterpreterExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> XlaInterpreterPlatform::ExecutorForDevice(
    int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*>
XlaInterpreterPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> XlaInterpreterPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
XlaInterpreterPlatform::GetUncachedExecutor(
    const StreamExecutorConfig& config) {
  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<XlaInterpreterExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

void XlaInterpreterPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register executor trace listener";
}

void XlaInterpreterPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister executor trace listener";
}

static void InitializeXlaInterpreterPlatform() {
  std::unique_ptr<Platform> platform(new XlaInterpreterPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace interpreter
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    interpreter_platform,
    stream_executor::interpreter::InitializeXlaInterpreterPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(interpreter_platform,
                                     multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     interpreter_platform);
