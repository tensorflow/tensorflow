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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "tensorflow/stream_executor/timer.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {

class TpuExecutor : public tensorflow::tpu::TpuExecutorInterface {
 public:
  using Status = ::stream_executor::port::Status;
  template <typename T>
  using StatusOr = ::stream_executor::port::StatusOr<T>;
  using StatusCallback = std::function<void(const Status&)>;
  using Stream = ::stream_executor::Stream;
  using Event = ::stream_executor::Event;
  using Timer = ::stream_executor::Timer;
  using DeviceMemoryBase = ::stream_executor::DeviceMemoryBase;
  using StreamInterface = ::stream_executor::internal::StreamInterface;
  using StreamExecutorInterface =
      ::stream_executor::internal::StreamExecutorInterface;

  using TimerMap =
      absl::flat_hash_map<stream_executor::internal::TimerInterface*,
                          SE_Timer*>;

  explicit TpuExecutor(::tensorflow::tpu::TpuPlatformInterface* platform,
                       SE_StreamExecutor* executor)
      : platform_(platform), executor_(executor) {}

  ~TpuExecutor() override;

  Status Init(int device_ordinal,
              ::stream_executor::DeviceOptions device_options) override;

  DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override;

  StatusOr<DeviceMemoryBase> AllocateDeviceMemoryBase(uint64 size,
                                                      int64 memory_space);

  Status AllocateEvent(Event* event) override;

  bool AllocateStream(Stream* stream) override;

  bool AllocateTimer(Timer* timer) override;

  Status BlockHostUntilDone(::stream_executor::Stream* stream) override;

  Status BlockUntilDoneOrFailed();

  StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
  CreateDeviceDescription() const override;

  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  void DeallocateStream(Stream* stream) override;

  void Deallocate(const DeviceMemoryBase& memory);

  void Deallocate(DeviceMemoryBase* memory) override;

  Status DeallocateEvent(Event* event) override;

  void DeallocateTimer(Timer* timer) override;

  bool DeviceMemoryUsage(int64* free, int64* total) const override;

  void DequeueOutfeed(int32 outfeed_queue_index, absl::Span<uint8> bytes,
                      StatusCallback done);

  Status EnqueueInfeed(int32 infeed_queue_index,
                       absl::Span<const uint8> bytes);

  absl::optional<stream_executor::AllocatorStats> GetAllocatorStats() override;

  Status GetStatus(Stream* stream) override;

  std::unique_ptr<::stream_executor::internal::StreamInterface>
  GetStreamImplementation() override;

  std::unique_ptr<::stream_executor::internal::TimerInterface>
  GetTimerImplementation() override;

  std::unique_ptr<::stream_executor::internal::EventInterface>
  CreateEventImplementation() override;

  bool HostCallback(Stream* stream, std::function<Status()> callback) override;

  bool Memcpy(Stream* stream, void* host_dst,
              const ::stream_executor::DeviceMemoryBase& device_src,
              uint64 size) override;

  bool Memcpy(Stream* stream, ::stream_executor::DeviceMemoryBase* device_dst,
              const void* host_src, uint64 size) override;

  bool MemcpyDeviceToDevice(Stream* stream,
                            ::stream_executor::DeviceMemoryBase* gpu_dst,
                            const ::stream_executor::DeviceMemoryBase& host_src,
                            uint64 size) override;

  void SyncAndForgetFailedStreams();
  bool SynchronizeAllActivity() override;

  Status SynchronousMemcpy(::stream_executor::DeviceMemoryBase* device_dst,
                           const void* host_src, uint64 size) override;
  Status SynchronousMemcpy(
      void* host_dst, const ::stream_executor::DeviceMemoryBase& device_src,
      uint64 size) override;
  Status SynchronousMemcpyDeviceToDevice(
      ::stream_executor::DeviceMemoryBase* device_dst,
      const ::stream_executor::DeviceMemoryBase& device_src,
      uint64 size) override;

  int PlatformDeviceCount() override;

  Event::Status PollForEventStatus(Event* event) override;
  Status RecordEvent(Stream* stream, ::stream_executor::Event* event) override;
  Status WaitForEvent(Stream* stream, ::stream_executor::Event* event) override;

  bool StartTimer(Stream* stream, ::stream_executor::Timer* timer) override;
  bool StopTimer(Stream* stream, ::stream_executor::Timer* timer) override;

  Status WaitForInfeedReady(int32 infeed_queue_index);

  Status WaitForOutfeedReady(int32 outfeed_queue_index);

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
  CreateTemporaryDeviceMemory(int64 memory_space, int64 byte_offset,
                              int64 size) override {
    LOG(FATAL) << "Unimplemented.";
  }

  // -- Unimplemented (stubbed out) methods.
  std::unique_ptr<stream_executor::internal::KernelInterface>
  CreateKernelImplementation() override {
    LOG(FATAL) << "Not yet implemented";
  }

  stream_executor::SharedMemoryConfig GetDeviceSharedMemoryConfig() override {
    LOG(FATAL) << "not yet implemented";
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset,
                     uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }
  Status MemZero(Stream* stream, DeviceMemoryBase* location,
                 uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }
  Status Memset32(Stream* stream, DeviceMemoryBase* location, uint32 pattern,
                  uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }
  Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    LOG(FATAL) << "not yet implemented";
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    LOG(FATAL) << "not yet implemented";
  }
  Status SetDeviceSharedMemoryConfig(
      stream_executor::SharedMemoryConfig config) override {
    LOG(FATAL) << "not yet implemented";
  }
  void* HostMemoryAllocate(uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }
  void HostMemoryDeallocate(void* mem) override {
    LOG(FATAL) << "not yet implemented";
  }
  bool HostMemoryRegister(void* mem, uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }
  bool HostMemoryUnregister(void* mem) override {
    LOG(FATAL) << "not yet implemented";
  }
  Status SynchronousMemZero(DeviceMemoryBase* location, uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }
  Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                           uint64 size) override {
    LOG(FATAL) << "not yet implemented";
  }

 private:
  TimerMap timer_map_;

  TpuPlatform::StreamMap& stream_map() {
    return *(static_cast<TpuPlatform*>(platform_)->stream_map());
  }

  TpuPlatform::EventMap& event_map() {
    return *(static_cast<TpuPlatform*>(platform_)->event_map());
  }

  ::tensorflow::tpu::TpuPlatformInterface* platform_;
  SE_StreamExecutor* executor_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
