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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"  // IWYU pragma: keep
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "xla/stream_executor/trace_listener.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tpu {

class TpuPlatform : public ::tensorflow::tpu::TpuPlatformInterface {
 public:
  using StreamMap =
      absl::flat_hash_map<stream_executor::internal::StreamInterface*,
                          SE_Stream*>;
  using EventMap =
      absl::flat_hash_map<stream_executor::internal::EventInterface*,
                          SE_Event*>;

  static const ::stream_executor::Platform::Id kId;

  template <typename T>
  using StatusOr = ::tsl::StatusOr<T>;

  TpuPlatform();

  ~TpuPlatform() override;

  static TpuPlatform* GetRegisteredPlatform();

  Id id() const override;

  const std::string& Name() const override;

  int VisibleDeviceCount() const override;

  bool ShouldRegisterTpuDeviceToDeviceCopy() override;

  const tensorflow::tpu::TpuTopologyPtr GetTopologyPtr() override;

  const tensorflow::tpu::TpuHostLocationExternal GetTpuHostLocation()
      const override;

  TpuRuntimeVersion version() const override;

  bool Initialized() const override;

  tsl::Status Initialize(
      const std::map<std::string, std::string>& platform_options) override;

  tsl::Status Reset(bool only_tear_down, absl::string_view reason) override {
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

  StatusOr<::stream_executor::StreamExecutor*> GetExecutor(
      const ::stream_executor::StreamExecutorConfig& config) override;

  StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
  GetUncachedExecutor(
      const ::stream_executor::StreamExecutorConfig& config) override;

  StreamMap* stream_map() { return &stream_map_; }

  void InsertEvent(stream_executor::internal::EventInterface* key,
                   SE_Event* val);
  SE_Event* LookupEvent(stream_executor::internal::EventInterface* key);
  SE_Stream* LookupStream(stream_executor::internal::StreamInterface* key) {
    mutex().Lock();
    auto stream = stream_map_.at(key);
    mutex().Unlock();
    return stream;
  }
  void EraseEvent(stream_executor::internal::EventInterface* key);

  SE_Platform* se_platform() const { return platform_; }

  // Returns the number of TPUs per host.
  static tsl::Status TpusPerHost(int* tpus);

  // Returns the memory capacity of the TPUs on this host.
  static tsl::Status TpuMemoryLimit(int64_t* memory_limit);

  absl::Mutex& mutex() { return event_map_mu_; }

 private:
  mutable SE_Platform* platform_;
  std::string name_;
  stream_executor::ExecutorCache executor_cache_;
  StreamMap stream_map_;
  EventMap event_map_;
  absl::Mutex event_map_mu_;
};

bool RegisterTpuPlatform();

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
