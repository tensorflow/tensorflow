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

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

#include <poplar/Device.hpp>

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace perftools {
namespace gputools {
namespace poplarplugin {

PoplarPlatform::PoplarPlatform() : name_("Poplar"), num_devices_(1) {
  /*
  auto device_set = poplar::DeviceSet::getDeviceSet();
  if (device_set.getDevices().size() > 0) {
    num_devices_ = device_set.getDevices().size();
    if (num_devices_ > 4) {
      num_devices_ = 4;
    }
  }
  */
}

PoplarPlatform::~PoplarPlatform() {}

Platform::Id PoplarPlatform::id() const { return kPoplarPlatformId; }

int PoplarPlatform::VisibleDeviceCount() const {
  return num_devices_;
}

const string& PoplarPlatform::Name() const { return name_; }

port::StatusOr<StreamExecutor*> PoplarPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> PoplarPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> PoplarPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
PoplarPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = port::MakeUnique<StreamExecutor>(
      this, port::MakeUnique<PoplarExecutor>());
  TF_RETURN_IF_ERROR(executor->Init(config.ordinal, config.device_options));

  return std::move(executor);
}

void PoplarPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register poplar trace listener";
}

void PoplarPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister poplar trace listener";
}

port::Status
PoplarPlatform::ConfigurePoplarDevices(const tensorflow::IPUOptions& opts) {

  for (int ordinal = 0; ordinal < VisibleDeviceCount(); ordinal++) {
    StreamExecutor *executor;
    TF_ASSIGN_OR_RETURN(executor, ExecutorForDevice(ordinal));

    auto *e = static_cast<PoplarExecutor *>(executor->implementation());

    if (opts.device_config().size() > ordinal) {
      TF_RETURN_IF_ERROR(
          e->InitializePoplarDevice(ordinal, opts.device_config(ordinal)));
    } else {
      tensorflow::IPUOptions::DeviceConfig default_config;
      TF_RETURN_IF_ERROR(e->InitializePoplarDevice(ordinal, default_config));
    }
  }

  return port::Status::OK();
}

port::Status PoplarPlatform::GetCompilerReports(std::string& out) {
  for (int ordinal = 0; ordinal < VisibleDeviceCount(); ordinal++) {
    StreamExecutor* executor;
    TF_ASSIGN_OR_RETURN(executor, ExecutorForDevice(ordinal));

    auto* e = static_cast<PoplarExecutor*>(executor->implementation());

    std::string reports;
    TF_RETURN_IF_ERROR(e->GetCompilerReports(reports));

    out += reports;
  }

  return port::Status::OK();
}

static void InitializePoplarPlatform() {
  std::unique_ptr<se::Platform> platform(new sep::PoplarPlatform);
  SE_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace poplarplugin
}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(
    poplar_platform, sep::InitializePoplarPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(poplar_platform, multi_platform_manager);
