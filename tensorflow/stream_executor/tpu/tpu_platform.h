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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {

class TpuPlatform : public ::tensorflow::tpu::TpuPlatformInterface {
 public:
  using StreamMap =
      absl::flat_hash_map<stream_executor::internal::StreamInterface*,
                          SE_Stream*>;
  using EventMap =
      absl::flat_hash_map<stream_executor::internal::EventInterface*,
                          SE_Event*>;

  static const ::stream_executor::Platform::Id kId;
  static constexpr char kName[] = "TPU";

  using Status = ::stream_executor::port::Status;
  template <typename T>
  using StatusOr = ::stream_executor::port::StatusOr<T>;

  TpuPlatform();

  ~TpuPlatform() override;

  static TpuPlatform* GetRegisteredPlatform();

  Id id() const override;

  const std::string& Name() const override;

  int VisibleDeviceCount() const override;

  int64 TpuMemoryLimit() override;

  bool ShouldRegisterTpuDeviceToDeviceCopy() override;

  bool Initialized() const override {
    return TpuPlatform_Initialized(platform_);
  }

  Status Initialize(
      const std::map<std::string, std::string>& platform_options) override;

  Status Reset() override { return Reset(false); }

  Status Reset(bool only_tear_down) override {
    LOG(FATAL) << "Not yet implemented";
  }

  StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
  DescriptionForDevice(int ordinal) const override {
    LOG(FATAL) << "Not yet implemented";
  }

  StatusOr<::stream_executor::StreamExecutor*> ExecutorForDevice(
      int ordinal) override {
    stream_executor::StreamExecutorConfig config;
    config.ordinal = ordinal;
    return GetExecutor(config);
  }

  StatusOr<::stream_executor::StreamExecutor*>
  ExecutorForDeviceWithPluginConfig(
      int ordinal,
      const ::stream_executor::PluginConfig& plugin_config) override {
    stream_executor::StreamExecutorConfig config;
    config.ordinal = ordinal;
    config.plugin_config = plugin_config;
    return GetExecutor(config);
  }

  StatusOr<::stream_executor::StreamExecutor*> GetExecutor(
      const ::stream_executor::StreamExecutorConfig& config) override;

  StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
  GetUncachedExecutor(
      const ::stream_executor::StreamExecutorConfig& config) override;

  void RegisterTraceListener(
      std::unique_ptr<stream_executor::TraceListener> listener) override {
    LOG(FATAL) << "Not yet implemented";
  }

  void UnregisterTraceListener(
      stream_executor::TraceListener* listener) override {
    LOG(FATAL) << "Not yet implemented";
  }

  StreamMap* stream_map() { return &stream_map_; }

  EventMap* event_map() { return &event_map_; }

 private:
  SE_Platform* platform_;

  stream_executor::ExecutorCache executor_cache_;
  StreamMap stream_map_;
  EventMap event_map_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
