/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/host_callback_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
namespace gpu {

class HostCallbackRegistry;

class CudaStream : public StreamCommon {
 public:
  absl::Status WaitFor(Stream* other) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status WaitFor(Event* event) override;

  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src, uint64_t size) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback,
      absl::AnyInvocable<void(absl::Status) &&> error_cb) override;
  absl::Status BlockHostUntilDone() override;
  absl::Status RefreshStatus() override;

  void SetName(std::string name) override;

  Stream::PlatformSpecificHandle platform_specific_handle() const override {
    return {stream_handle_};
  }

  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      bool use_delay_kernel) override {
    return executor_->CreateEventBasedTimer(this, use_delay_kernel);
  }

  static absl::StatusOr<std::unique_ptr<CudaStream>> Create(
      CudaExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);

  ~CudaStream() override;

  CUstream stream_handle() const { return stream_handle_; }

 private:
  CudaStream(CudaExecutor* executor, CudaEvent completed_event,
             std::optional<std::variant<StreamPriority, int>> priority,
             CUstream stream_handle);

  absl::Status RecordCompletedEvent();

  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name, void** args,
                            int64_t shmem_bytes, bool use_pdl) override;

  StreamExecutor* executor_;
  CudaEvent completed_event_;
  CUstream stream_handle_;
  std::atomic<uint32_t> tsan_proxy_{false};
  std::unique_ptr<HostCallbackRegistry::RegistryHandle>
      callback_registry_handle_;
};
}  // namespace gpu

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_
