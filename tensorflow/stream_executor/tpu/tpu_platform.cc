/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/tpu/tpu_platform.h"

#include "tensorflow/c/tf_status.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

namespace tensorflow {

PLATFORM_DEFINE_ID(TpuPlatform::kId);
TpuPlatform* tpu_registered_platform = nullptr;

using Status = ::stream_executor::port::Status;
template <typename T>
using StatusOr = ::stream_executor::port::StatusOr<T>;

TpuPlatform::TpuPlatform() { platform_ = TpuPlatform_New(); }

TpuPlatform* TpuPlatform::GetRegisteredPlatform() {
  return tpu_registered_platform;
}

Status TpuPlatform::Initialize(
    const std::map<std::string, std::string>& platform_options) {
  StatusHelper status;

  size_t options_size = platform_options.size();
  const char** options_key =
      static_cast<const char**>(malloc(sizeof(const char*) * options_size));
  const char** options_value =
      static_cast<const char**>(malloc(sizeof(const char*) * options_size));

  size_t i = 0;
  for (const auto& option : platform_options) {
    options_key[i] = option.first.c_str();
    options_value[i] = option.second.c_str();
    i++;
  }

  TpuPlatform_Initialize(platform_, options_size, options_key, options_value,
                         status.c_status);

  free(options_key);
  free(options_value);

  return status.status();
}

TpuPlatform::~TpuPlatform() { TpuPlatform_Free(platform_); }

int TpuPlatform::VisibleDeviceCount() const {
  return TpuPlatform_VisibleDeviceCount(platform_);
}

StatusOr<::stream_executor::StreamExecutor*> TpuPlatform::GetExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
TpuPlatform::GetUncachedExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
  SE_StreamExecutorConfig* c_config = TpuStreamExecutorConfig_Default();

  TpuStreamExecutorConfig_SetOrdinal(c_config, config.ordinal);

  StatusHelper status;
  SE_StreamExecutor* executor =
      TpuPlatform_GetExecutor(platform_, c_config, status.c_status);
  TpuStreamExecutorConfig_Free(c_config);
  if (!status.ok()) {
    return status.status();
  }
  return std::make_unique<stream_executor::StreamExecutor>(
      this, absl::make_unique<tensorflow::TpuExecutor>(this, executor),
      config.ordinal);
}

::stream_executor::Platform::Id TpuPlatform::id() const {
  return TpuPlatform::kId;
}

const std::string& TpuPlatform::Name() const {
  static std::string* name = new std::string(kName);
  return *name;
}

int64 TpuPlatform::TpuMemoryLimit() {
  return TpuPlatform_TpuMemoryLimit(platform_);
}

bool TpuPlatform::ShouldRegisterTpuDeviceToDeviceCopy() {
  return TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy(platform_);
}

}  // namespace tensorflow

void RegisterTpuPlatform() {
  tensorflow::tpu_registered_platform = new tensorflow::TpuPlatform();
  std::unique_ptr<stream_executor::Platform> platform(
      tensorflow::tpu_registered_platform);
  SE_CHECK_OK(stream_executor::MultiPlatformManager::RegisterPlatform(
      std::move(platform)));
}

REGISTER_MODULE_INITIALIZER(tpu_platform, RegisterTpuPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(tpu_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     tpu_platform);
