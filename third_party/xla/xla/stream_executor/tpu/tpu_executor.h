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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_options.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"
#include "xla/stream_executor/tpu/tpu_executor_interface.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/types.h"

namespace stream_executor {
namespace tpu {

class TpuExecutor : public tensorflow::tpu::TpuExecutorInterface {
 public:
  template <typename T>
  using StatusOr = ::tsl::StatusOr<T>;
  using StatusCallback = std::function<void(const tsl::Status&)>;
  using Stream = ::stream_executor::Stream;
  using Event = ::stream_executor::Event;
  using DeviceMemoryBase = ::stream_executor::DeviceMemoryBase;
  using StreamInterface = ::stream_executor::internal::StreamInterface;
  using StreamExecutorInterface =
      ::stream_executor::internal::StreamExecutorInterface;

  explicit TpuExecutor(::tensorflow::tpu::TpuPlatformInterface* platform,
                       SE_StreamExecutor* executor)
      : platform_(platform), executor_(executor) {}

  ~TpuExecutor() override;

  tsl::Status Init(int device_ordinal,
                   ::stream_executor::DeviceOptions device_options) override;

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;

  tsl::Status AllocateEvent(Event* event) override;

  bool AllocateStream(Stream* stream) override;

  tsl::Status BlockHostUntilDone(::stream_executor::Stream* stream) override;

  StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
  CreateDeviceDescription() const override;

  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  void DeallocateStream(Stream* stream) override;

  void Deallocate(const DeviceMemoryBase& memory);

  void Deallocate(DeviceMemoryBase* memory) override;

  tsl::Status DeallocateEvent(Event* event) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  void DequeueOutfeed(int32_t outfeed_queue_index, absl::Span<uint8_t> bytes,
                      StatusCallback done);

  tsl::Status EnqueueInfeed(int32_t infeed_queue_index,
                            absl::Span<const uint8_t> bytes);

  std::optional<stream_executor::AllocatorStats> GetAllocatorStats() override;

  tensorflow::tpu::TpuCoreLocationExternal GetCoreLocationExternal()
      const override;

  tsl::Status GetStatus(Stream* stream) override;

  std::unique_ptr<::stream_executor::internal::StreamInterface>
  GetStreamImplementation() override;

  std::unique_ptr<::stream_executor::internal::EventInterface>
  CreateEventImplementation() override;

  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;

  bool Memcpy(Stream* stream, void* host_dst,
              const ::stream_executor::DeviceMemoryBase& device_src,
              uint64_t size) override;

  bool Memcpy(Stream* stream, ::stream_executor::DeviceMemoryBase* device_dst,
              const void* host_src, uint64_t size) override;

  bool MemcpyDeviceToDevice(Stream* stream,
                            ::stream_executor::DeviceMemoryBase* gpu_dst,
                            const ::stream_executor::DeviceMemoryBase& host_src,
                            uint64_t size) override;

  bool SynchronizeAllActivity() override;

  tsl::Status SynchronousMemcpy(::stream_executor::DeviceMemoryBase* device_dst,
                                const void* host_src, uint64_t size) override;
  tsl::Status SynchronousMemcpy(
      void* host_dst, const ::stream_executor::DeviceMemoryBase& device_src,
      uint64_t size) override;
  tsl::Status SynchronousMemcpyDeviceToDevice(
      ::stream_executor::DeviceMemoryBase* device_dst,
      const ::stream_executor::DeviceMemoryBase& device_src,
      uint64_t size) override;

  Event::Status PollForEventStatus(Event* event) override;
  tsl::Status RecordEvent(Stream* stream,
                          ::stream_executor::Event* event) override;
  tsl::Status WaitForEvent(Stream* stream,
                           ::stream_executor::Event* event) override;

  tsl::Status UnloadAllPrograms() override;

  tsl::Status EnqueueCompactionOnStreamForHbm(
      Stream* compaction_stream) override;

  const ::tensorflow::tpu::TpuPlatformInterface& platform() const override {
    return *platform_;
  }

  ::tensorflow::tpu::TpuPlatformInterface& platform() override {
    return *platform_;
  }

  // TODO(henrytan): convert this to override once the base interface is changed
  // to TpuExecutorInterface.
  StatusOr<std::unique_ptr<
      tensorflow::tpu::TpuExecutorInterface::TemporaryDeviceMemory>>
  CreateTemporaryDeviceMemory(int64_t memory_space, int64_t byte_offset,
                              int64_t size) override {
    LOG(FATAL) << "Unimplemented.";
  }

  // -- Unimplemented (stubbed out) methods.
  std::unique_ptr<stream_executor::internal::KernelInterface>
  CreateKernelImplementation() override {
    LOG(FATAL) << "Not yet implemented";
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                     uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }
  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                      uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                       uint32_t pattern, uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }
  tsl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    LOG(FATAL) << "not yet implemented";
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    LOG(FATAL) << "not yet implemented";
  }

  void* HostMemoryAllocate(uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }
  void HostMemoryDeallocate(void* mem) override {
    LOG(FATAL) << "not yet implemented";
  }
  bool HostMemoryRegister(void* mem, uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }
  bool HostMemoryUnregister(void* mem) override {
    LOG(FATAL) << "not yet implemented";
  }
  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                 uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }
  tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                uint64_t size) override {
    LOG(FATAL) << "not yet implemented";
  }

  SE_StreamExecutor* se_executor() { return executor_; }

 private:
  tensorflow::tpu::TpuPlatform& tpu_platform() {
    return *(tensorflow::down_cast<tensorflow::tpu::TpuPlatform*>(platform_));
  }

  tensorflow::tpu::TpuPlatform::StreamMap& stream_map() {
    return *(tpu_platform().stream_map());
  }

  SE_Stream* get_stream(StreamInterface* ptr) {
    absl::MutexLock m(&tpu_platform().mutex());
    return stream_map()[ptr];
  }

  tensorflow::tpu::TpuPlatformInterface* platform_;
  SE_StreamExecutor* executor_;
};

}  // namespace tpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
