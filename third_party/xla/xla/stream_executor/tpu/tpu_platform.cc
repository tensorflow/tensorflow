/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/tpu_platform.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_interface.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_executor.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_platform_id.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {
namespace tpu {

const ::stream_executor::Platform::Id TpuPlatform::kId = GetTpuPlatformId();
TpuPlatform* tpu_registered_platform = nullptr;

template <typename T>
using StatusOr = ::absl::StatusOr<T>;

TpuPlatform::TpuPlatform() : name_("TPU") {
  platform_ = stream_executor::tpu::ExecutorApiFn()->TpuPlatform_NewFn();
  CHECK(platform_ != nullptr);
}

TpuPlatform* TpuPlatform::GetRegisteredPlatform() {
  return tpu_registered_platform;
}

absl::Status TpuPlatform::Initialize(
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

  stream_executor::tpu::ExecutorApiFn()->TpuPlatform_InitializeFn(
      platform_, options_size, options_key, options_value, status.c_status);

  free(options_key);
  free(options_value);

  return status.status();
}

bool TpuPlatform::Initialized() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuPlatform_InitializedFn(
      platform_);
}

TpuPlatform::~TpuPlatform() {
  stream_executor::tpu::ExecutorApiFn()->TpuPlatform_FreeFn(platform_);
}

int TpuPlatform::VisibleDeviceCount() const {
  return stream_executor::tpu::ExecutorApiFn()
      ->TpuPlatform_VisibleDeviceCountFn(platform_);
}

StatusOr<::stream_executor::StreamExecutor*> TpuPlatform::GetExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
TpuPlatform::GetUncachedExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
  SE_StreamExecutorConfig* c_config = stream_executor::tpu::ExecutorApiFn()
                                          ->TpuStreamExecutorConfig_DefaultFn();

  stream_executor::tpu::ExecutorApiFn()->TpuStreamExecutorConfig_SetOrdinalFn(
      c_config, config.ordinal);

  StatusHelper status;
  SE_StreamExecutor* executor =
      stream_executor::tpu::ExecutorApiFn()->TpuPlatform_GetExecutorFn(
          platform_, c_config, status.c_status);
  stream_executor::tpu::ExecutorApiFn()->TpuStreamExecutorConfig_FreeFn(
      c_config);
  if (!status.ok()) {
    return status.status();
  }
  return std::make_unique<stream_executor::StreamExecutor>(
      this, std::make_unique<stream_executor::tpu::TpuExecutor>(
                this, executor, config.ordinal));
}

::stream_executor::Platform::Id TpuPlatform::id() const {
  return TpuPlatform::kId;
}

const std::string& TpuPlatform::Name() const { return name_; }

bool TpuPlatform::ShouldRegisterTpuDeviceToDeviceCopy() {
  return stream_executor::tpu::ExecutorApiFn()
      ->TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopyFn(platform_);
}

const tensorflow::tpu::TpuTopologyPtr TpuPlatform::GetTopologyPtr() {
  return stream_executor::tpu::ExecutorApiFn()->TpuPlatform_GetTopologyPtrFn(
      platform_);
}

const tensorflow::tpu::TpuHostLocationExternal TpuPlatform::GetTpuHostLocation()
    const {
  return tpu::TpuHostLocationExternal(
      stream_executor::tpu::ExecutorApiFn()->TpuPlatform_GetHostLocationFn(
          platform_));
}

TpuRuntimeVersion TpuPlatform::version() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuPlatform_GetRuntimeVersionFn(
      platform_);
}

void TpuPlatform::InsertEvent(stream_executor::EventInterface* key,
                              SE_Event* val) {
  absl::MutexLock lock(&event_map_mu_);
  event_map_[key] = val;
}

SE_Event* TpuPlatform::LookupEvent(stream_executor::EventInterface* key) {
  absl::ReaderMutexLock lock(&event_map_mu_);
  return event_map_.at(key);
}

void TpuPlatform::EraseEvent(stream_executor::EventInterface* key) {
  absl::MutexLock lock(&event_map_mu_);
  event_map_.erase(key);
}

absl::Status TpuPlatform::TpusPerHost(int* tpus) {
  if (stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_TpusPerHostFn ==
      nullptr) {
    *tpus = 0;
    return absl::OkStatus();
  }

  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_TpusPerHostFn(
      tpus, status.c_status);
  return status.status();
}

absl::Status TpuPlatform::TpuMemoryLimit(int64_t* memory_limit) {
  if (stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_TpuMemoryLimitFn ==
      nullptr) {
    *memory_limit = 0;
    return absl::OkStatus();
  }

  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_TpuMemoryLimitFn(
      reinterpret_cast<int64_t*>(memory_limit), status.c_status);
  return status.status();
}

bool RegisterTpuPlatform() {
  // Silently bail if the underlying TPU C API isn't initialized. This is useful
  // for code that unconditionally calls RegisterTpuPlatform() but doesn't link
  // in the underlying TPU library when not running on TPU.
  if (!stream_executor::tpu::IsStreamExecutorEnabled(
          stream_executor::tpu::ExecutorApiFn())) {
    return true;
  }
  static bool tpu_platform_registered = false;
  if (!tpu_platform_registered) {
    tpu_registered_platform = new TpuPlatform();
    std::unique_ptr<stream_executor::Platform> platform(
        tpu_registered_platform);
    TF_CHECK_OK(stream_executor::PlatformManager::RegisterPlatform(
        std::move(platform)));
    tpu_platform_registered = true;
  }
  return true;
}

}  // namespace tpu
}  // namespace tensorflow
