/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"

namespace stream_executor {
namespace host {

// An implementation of StreamExecutor that does no communication or interaction
// with a device, but DOES perform memory operations backed by the host.
// Plugin routines (BLAS) are also supported and functional.
// Kernel invocations will fail, but host callbacks may be enqueued on this
// executor and its associated stream, and should follow standard ordering
// semantics.
//
// This is useful for evaluating the performance of host-based or fallback
// routines executed under the context of a GPU executor.
// See stream_executor.h for description of the below operations.
class HostExecutor : public internal::StreamExecutorInterface {
 public:
  HostExecutor() = default;

  // The stack size used for host streams can be set via
  // device_options.non_portable_tags["host_stack_size"].
  tsl::Status Init(int device_ordinal, DeviceOptions device_options) override;

  tsl::Status GetKernel(const MultiKernelLoaderSpec& spec,
                        Kernel* kernel) override {
    return tsl::errors::Unimplemented("Not Implemented");
  }
  tsl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                     const BlockDim& block_dims, const Kernel& kernel,
                     const KernelArgs& args) override {
    return tsl::errors::Unimplemented("Not Implemented");
  }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset_bytes,
                     uint64_t size_bytes) override;
  void Deallocate(DeviceMemoryBase* mem) override;

  void* HostMemoryAllocate(uint64_t size) override { return new char[size]; }
  void HostMemoryDeallocate(void* mem) override {
    delete[] static_cast<char*>(mem);
  }
  bool HostMemoryRegister(void* mem, uint64_t size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override;
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;

  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                      uint64_t size) override;
  tsl::Status Memset(Stream* stream, DeviceMemoryBase* location,
                     uint8_t pattern, uint64_t size) override;
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                       uint32_t pattern, uint64_t size) override;

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override { return true; }
  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                 uint64_t size) override;

  tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                uint64_t size) override;

  tsl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                              const DeviceMemoryBase& gpu_src,
                                              uint64_t size) override;

  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;

  tsl::Status AllocateEvent(Event* event) override;
  tsl::Status DeallocateEvent(Event* event) override;
  tsl::Status RecordEvent(Stream* stream, Event* event) override;
  tsl::Status WaitForEvent(Stream* stream, Event* event) override;
  Event::Status PollForEventStatus(Event* event) override;

  bool AllocateStream(Stream* stream) override;
  void DeallocateStream(Stream* stream) override;
  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  tsl::Status BlockHostUntilDone(Stream* stream) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(0);
  }

  static tsl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  tsl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return ::tsl::OkStatus();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return true;
  }

  blas::BlasSupport* CreateBlas() override;

  dnn::DnnSupport* CreateDnn() override { return nullptr; }

  fft::FftSupport* CreateFft() override;

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

 private:
  // Size of thread stacks for streams in bytes. '0' means "the default size".
  size_t thread_stack_size_in_bytes_ = 0;
};

}  // namespace host
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
