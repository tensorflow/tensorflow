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
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_id.h"

namespace tensorflow {
namespace tpu {

const ::stream_executor::Platform::Id TpuPlatform::kId = GetTpuPlatformId();
TpuPlatform* tpu_registered_platform = nullptr;

using Status = ::stream_executor::port::Status;
template <typename T>
using StatusOr = ::stream_executor::port::StatusOr<T>;

TpuPlatform::TpuPlatform() : name_("TPU") {
  platform_ = tpu::ExecutorApiFn()->TpuPlatform_NewFn();
  CHECK(platform_ != nullptr);
}

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

  tpu::ExecutorApiFn()->TpuPlatform_InitializeFn(
      platform_, options_size, options_key, options_value, status.c_status);

  free(options_key);
  free(options_value);

  return status.status();
}

bool TpuPlatform::Initialized() const {
  return tpu::ExecutorApiFn()->TpuPlatform_InitializedFn(platform_);
}

TpuPlatform::~TpuPlatform() {
  tpu::ExecutorApiFn()->TpuPlatform_FreeFn(platform_);
}

int TpuPlatform::VisibleDeviceCount() const {
  return tpu::ExecutorApiFn()->TpuPlatform_VisibleDeviceCountFn(platform_);
}

StatusOr<::stream_executor::StreamExecutor*> TpuPlatform::GetExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
TpuPlatform::GetUncachedExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
  SE_StreamExecutorConfig* c_config =
      tpu::ExecutorApiFn()->TpuStreamExecutorConfig_DefaultFn();

  tpu::ExecutorApiFn()->TpuStreamExecutorConfig_SetOrdinalFn(c_config,
                                                             config.ordinal);

  StatusHelper status;
  SE_StreamExecutor* executor = tpu::ExecutorApiFn()->TpuPlatform_GetExecutorFn(
      platform_, c_config, status.c_status);
  tpu::ExecutorApiFn()->TpuStreamExecutorConfig_FreeFn(c_config);
  if (!status.ok()) {
    return status.status();
  }
  return std::make_unique<stream_executor::StreamExecutor>(
      this, std::make_unique<TpuExecutor>(this, executor), config.ordinal);
}

::stream_executor::Platform::Id TpuPlatform::id() const {
  return TpuPlatform::kId;
}

const std::string& TpuPlatform::Name() const { return name_; }

int64_t TpuPlatform::TpuMemoryLimit() {
  return tpu::ExecutorApiFn()->TpuPlatform_TpuMemoryLimitFn(platform_);
}

bool TpuPlatform::ShouldRegisterTpuDeviceToDeviceCopy() {
  return tpu::ExecutorApiFn()
      ->TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopyFn(platform_);
}

const tensorflow::tpu::TpuTopologyPtr TpuPlatform::GetTopologyPtr() {
  return tpu::ExecutorApiFn()->TpuPlatform_GetTopologyPtrFn(platform_);
}

const tensorflow::tpu::TpuHostLocationExternal TpuPlatform::GetTpuHostLocation()
    const {
  return tpu::TpuHostLocationExternal(
      tpu::ExecutorApiFn()->TpuPlatform_GetHostLocationFn(platform_));
}

TpuRuntimeVersion TpuPlatform::version() const {
  return tpu::ExecutorApiFn()->TpuPlatform_GetRuntimeVersionFn(platform_);
}

void TpuPlatform::InsertEvent(stream_executor::internal::EventInterface* key,
                              SE_Event* val) {
  tensorflow::mutex_lock lock(event_map_mu_);
  event_map_[key] = val;
}

SE_Event* TpuPlatform::LookupEvent(
    stream_executor::internal::EventInterface* key) {
  tensorflow::tf_shared_lock lock(event_map_mu_);
  return event_map_.at(key);
}

void TpuPlatform::EraseEvent(stream_executor::internal::EventInterface* key) {
  tensorflow::mutex_lock lock(event_map_mu_);
  event_map_.erase(key);
}

Status TpuPlatform::TpusPerHost(int* tpus) {
  TF_Status* status = TF_NewStatus();

  if (tpu::OpsApiFn()->TpuConfigurationApi_TpusPerHostFn == nullptr) {
    *tpus = 0;
    return OkStatus();
  }

  tpu::OpsApiFn()->TpuConfigurationApi_TpusPerHostFn(tpus, status);
  auto ret_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return ret_status;
}

Status TpuPlatform::TpuMemoryLimit(int64_t* memory_limit) {
  TF_Status* status = TF_NewStatus();

  if (tpu::OpsApiFn()->TpuConfigurationApi_TpuMemoryLimitFn == nullptr) {
    *memory_limit = 0;
    return OkStatus();
  }

  tpu::OpsApiFn()->TpuConfigurationApi_TpuMemoryLimitFn(
      reinterpret_cast<int64_t*>(memory_limit), status);
  auto ret_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return ret_status;
}

bool RegisterTpuPlatform() {
  // Silently bail if the underlying TPU C API isn't initialized. This is useful
  // for code that unconditionally calls RegisterTpuPlatform() but doesn't link
  // in the underlying TPU library when not running on TPU.
  if (!tpu::IsStreamExecutorEnabled(tpu::ExecutorApiFn())) {
    return true;
  }
  static bool tpu_platform_registered = false;
  if (!tpu_platform_registered) {
    tpu_registered_platform = new TpuPlatform();
    std::unique_ptr<stream_executor::Platform> platform(
        tpu_registered_platform);
    SE_CHECK_OK(stream_executor::MultiPlatformManager::RegisterPlatform(
        std::move(platform)));
    tpu_platform_registered = true;
  }
  return true;
}

}  // namespace tpu
}  // namespace tensorflow
