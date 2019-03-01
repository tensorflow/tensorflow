/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

PoplarPlatform::PoplarPlatform() : name_("Poplar") {
  VLOG(1) << "Poplar version: " << poplar::versionString();
}

PoplarPlatform::~PoplarPlatform() {}

se::Platform::Id PoplarPlatform::id() const { return kPoplarPlatformId; }

int PoplarPlatform::VisibleDeviceCount() const {
  int num_devices = PoplarExecutor::GetDeviceManager()
                        .getDevices(poplar::TargetType::IPU, 1)
                        .size();

  if (num_devices == 0) {
    // Allow for 2 virtual devices
    num_devices = 2;
  }

  return num_devices;
}

const std::string& PoplarPlatform::Name() const { return name_; }

StatusOr<se::StreamExecutor*> PoplarPlatform::ExecutorForDevice(int ordinal) {
  se::StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = se::PluginConfig();
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

StatusOr<se::StreamExecutor*> PoplarPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const se::PluginConfig& plugin_config) {
  se::StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

StatusOr<se::StreamExecutor*> PoplarPlatform::GetExecutor(
    const se::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

StatusOr<std::unique_ptr<se::StreamExecutor>>
PoplarPlatform::GetUncachedExecutor(const se::StreamExecutorConfig& config) {
  auto executor = absl::make_unique<se::StreamExecutor>(
      this, absl::make_unique<PoplarExecutor>());
  TF_RETURN_IF_ERROR(executor->Init(config.ordinal, config.device_options));

  return std::move(executor);
}

void PoplarPlatform::RegisterTraceListener(
    std::unique_ptr<se::TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register poplar trace listener";
}

void PoplarPlatform::UnregisterTraceListener(se::TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister poplar trace listener";
}

Status PoplarPlatform::ConfigurePoplarDevice(
    int ordinal, const tensorflow::IPUOptions& opts) {
  se::StreamExecutor* executor;
  TF_ASSIGN_OR_RETURN(executor, ExecutorForDevice(ordinal));

  auto* e = static_cast<PoplarExecutor*>(executor->implementation());

  TF_RETURN_IF_ERROR(e->ConfigurePoplarDevice(opts));

  return Status::OK();
}

Status PoplarPlatform::GetCompilerEvents(
    std::list<tensorflow::IpuTraceEvent>& out) {
  for (int ordinal = 0; ordinal < VisibleDeviceCount(); ordinal++) {
    se::StreamExecutor* executor;
    TF_ASSIGN_OR_RETURN(executor, ExecutorForDevice(ordinal));

    auto* e = static_cast<PoplarExecutor*>(executor->implementation());

    TF_RETURN_IF_ERROR(e->GetCompilerEvents(out));
  }

  return Status::OK();
}

void PoplarPlatform::AboutToFreeEngine(poplar::Engine* engine) {
  for (int ordinal = 0; ordinal < VisibleDeviceCount(); ordinal++) {
    auto executor = ExecutorForDevice(ordinal);

    if (executor.ok()) {
      auto* e =
          static_cast<PoplarExecutor*>(executor.ValueOrDie()->implementation());

      e->AboutToFreeEngine(engine);
    }
  }
}

void PoplarPlatform::ResetXfeedManagers() {
  for (int ordinal = 0; ordinal < VisibleDeviceCount(); ordinal++) {
    ResetXfeedManager(ordinal);
  }
}

static void InitializePoplarPlatform() {
  std::unique_ptr<se::Platform> platform(new PoplarPlatform);
  SE_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace poplarplugin
}  // namespace xla

REGISTER_MODULE_INITIALIZER(poplar_platform,
                            xla::poplarplugin::InitializePoplarPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(poplar_platform, multi_platform_manager);
