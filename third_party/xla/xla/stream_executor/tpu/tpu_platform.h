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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"  // IWYU pragma: keep
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {
namespace tpu {

class TpuPlatform : public ::tensorflow::tpu::TpuPlatformInterface {
 public:
  using StreamMap = absl::flat_hash_map<stream_executor::Stream*, SE_Stream*>;
  using EventMap = absl::flat_hash_map<stream_executor::Event*, SE_Event*>;

  static const ::stream_executor::Platform::Id kId;

  TpuPlatform();

  ~TpuPlatform() override;

  static TpuPlatform* GetRegisteredPlatform();

  Id id() const override;

  const std::string& Name() const override;

  int VisibleDeviceCount() const override;

  bool ShouldRegisterTpuDeviceToDeviceCopy() override;

  const SE_TpuTopology* GetTopologyPtr() override;

  const tensorflow::tpu::TpuHostLocationExternal GetTpuHostLocation()
      const override;

  TpuRuntimeVersion version() const override;

  bool Initialized() const override;

  absl::Status Initialize() override;

  absl::Status Reset(bool only_tear_down, absl::string_view reason) override {
    LOG(FATAL) << "Not yet implemented";
  }

  absl::StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
  DescriptionForDevice(int ordinal) const override {
    LOG(FATAL) << "Not yet implemented";
  }

  absl::StatusOr<::stream_executor::StreamExecutor*> ExecutorForDevice(
      int ordinal) override;

  absl::StatusOr<::stream_executor::StreamExecutor*> FindExisting(
      int ordinal) override {
    return executor_cache_.Get(ordinal);
  }

  StreamMap* stream_map() { return &stream_map_; }

  void InsertEvent(stream_executor::Event* key, SE_Event* val);
  SE_Event* LookupEvent(stream_executor::Event* key);
  SE_Stream* LookupStream(stream_executor::Stream* key) {
    mutex().Lock();
    auto stream = stream_map_.at(key);
    mutex().Unlock();
    return stream;
  }
  void EraseEvent(stream_executor::Event* key) override;

  SE_Platform* se_platform() const { return platform_; }

  // Returns the number of TPUs per host.
  static absl::Status TpusPerHost(int* tpus);

  // Returns the memory capacity of the TPUs on this host.
  static absl::Status TpuMemoryLimit(int64_t* memory_limit);

  absl::Mutex& mutex() { return event_map_mu_; }

 private:
  // Returns a device constructed with the ordinal without
  // looking in or storing to the Platform's executor cache.
  // Ownership IS transferred to the caller.
  absl::StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
  GetUncachedExecutor(int ordinal);

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
