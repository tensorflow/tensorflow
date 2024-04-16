/* Copyright 2016 The OpenXLA Authors.

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

// Declares the HostExecutor class, which is a CPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {
namespace host {

// An implementation of StreamExecutor that does no communication or interaction
// with a device, but DOES perform memory operations backed by the host.
// Kernel invocations will fail, but host callbacks may be enqueued on this
// executor and its associated stream, and should follow standard ordering
// semantics.
//
// This is useful for evaluating the performance of host-based or fallback
// routines executed under the context of a GPU executor.
// See stream_executor.h for description of the below operations.
class HostExecutor : public StreamExecutorInterface {
 public:
  explicit HostExecutor(int device_ordinal) : device_ordinal_(device_ordinal) {}

  absl::Status Init() override;

  absl::Status GetKernel(const MultiKernelLoaderSpec& spec,
                         Kernel* kernel) override {
    return absl::UnimplementedError("Not Implemented");
  }
  absl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims, const Kernel& kernel,
                      const KernelArgs& args) override {
    return absl::UnimplementedError("Not Implemented");
  }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceMemoryBase* mem) override;

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override {
    return std::make_unique<HostMemoryAllocation>(new char[size], size, this);
  }
  void HostMemoryDeallocate(void* mem) override {
    delete[] static_cast<char*>(mem);
  }

  absl::Status Memcpy(Stream* stream, void* host_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) override;
  absl::Status Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                      const void* host_src, uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;

  absl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64_t size) override;
  absl::Status Memset(Stream* stream, DeviceMemoryBase* location,
                      uint8_t pattern, uint64_t size) override;
  absl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32_t pattern, uint64_t size) override;

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override { return true; }
  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;

  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;

  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<absl::Status() &&> callback) override;

  absl::Status AllocateEvent(Event* event) override;
  absl::Status DeallocateEvent(Event* event) override;
  absl::Status RecordEvent(Stream* stream, Event* event) override;
  absl::Status WaitForEvent(Stream* stream, Event* event) override;
  Event::Status PollForEventStatus(Event* event) override;

  bool AllocateStream(Stream* stream) override;
  void DeallocateStream(Stream* stream) override;
  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  absl::Status BlockHostUntilDone(Stream* stream) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(0);
  }

  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);
  int device_ordinal() const override { return device_ordinal_; }

  absl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return absl::OkStatus();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return true;
  }

  std::unique_ptr<EventInterface> CreateEventImplementation() override;

  std::unique_ptr<StreamInterface> GetStreamImplementation() override;

 private:
  int device_ordinal_;
};

}  // namespace host
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_EXECUTOR_H_
