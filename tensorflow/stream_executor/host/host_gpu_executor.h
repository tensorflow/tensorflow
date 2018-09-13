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
#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace host {

// An implementation of StreamExecutor that does no communication or interaction
// with a device, but DOES perform memory operations backed by the host.
// Plugin routines (RNG, BLAS) are also supported and functional.
// Kernel invocations will fail, but host callbacks may be enqueued on this
// executor and its associated stream, and should follow standard ordering
// semantics.
//
// This is useful for evaluating the performance of host-based or fallback
// routines executed under the context of a GPU executor.
// See stream_executor.h for description of the below operations.
class HostExecutor : public internal::StreamExecutorInterface {
 public:
  explicit HostExecutor(const PluginConfig &plugin_config);
  ~HostExecutor() override;

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
    return port::Status::OK();
  }

  bool GetKernel(const MultiKernelLoaderSpec &spec,
                 KernelBase *kernel) override {
    return false;
  }
  bool Launch(Stream *stream, const ThreadDim &thread_dims,
              const BlockDim &block_dims, const KernelBase &kernel,
              const KernelArgsArrayBase &args) override {
    return false;
  }

  void *Allocate(uint64 size) override;
  void *AllocateSubBuffer(DeviceMemoryBase *mem, uint64 offset_bytes,
                          uint64 size_bytes) override;
  void Deallocate(DeviceMemoryBase *mem) override;

  void *HostMemoryAllocate(uint64 size) override { return new char[size]; }
  void HostMemoryDeallocate(void *mem) override {
    delete[] static_cast<char *>(mem);
  }
  bool HostMemoryRegister(void *mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void *mem) override { return true; }

  bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &gpu_src,
              uint64 size) override;
  bool Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst, const void *host_src,
              uint64 size) override;
  bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *gpu_dst,
                            const DeviceMemoryBase &host_src,
                            uint64 size) override;

  bool MemZero(Stream *stream, DeviceMemoryBase *location,
               uint64 size) override;
  bool Memset(Stream *stream, DeviceMemoryBase *location, uint8 pattern,
              uint64 size) override;
  bool Memset32(Stream *stream, DeviceMemoryBase *location, uint32 pattern,
                uint64 size) override;

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override { return true; }
  bool SynchronousMemZero(DeviceMemoryBase *location, uint64 size) override;

  bool SynchronousMemSet(DeviceMemoryBase *location, int value,
                         uint64 size) override;

  port::Status SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                 const void *host_src, uint64 size) override;
  port::Status SynchronousMemcpy(void *host_dst,
                                 const DeviceMemoryBase &gpu_src,
                                 uint64 size) override;
  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *gpu_dst,
                                               const DeviceMemoryBase &gpu_src,
                                               uint64 size) override;

  bool HostCallback(Stream *stream, std::function<void()> callback) override;

  port::Status AllocateEvent(Event *event) override {
    return port::Status(port::error::UNIMPLEMENTED, "");
  }

  port::Status DeallocateEvent(Event *event) override {
    return port::Status(port::error::UNIMPLEMENTED, "");
  }

  port::Status RecordEvent(Stream *stream, Event *event) override {
    return port::Status(port::error::UNIMPLEMENTED, "");
  }

  port::Status WaitForEvent(Stream *stream, Event *event) override {
    return port::Status(port::error::UNIMPLEMENTED, "");
  }

  Event::Status PollForEventStatus(Event *event) override {
    return Event::Status::kError;
  }

  bool AllocateStream(Stream *stream) override;
  void DeallocateStream(Stream *stream) override;
  bool CreateStreamDependency(Stream *dependent, Stream *other) override;

  // No special initialization is necessary for host timers.
  bool AllocateTimer(Timer *timer) override { return true; }

  void DeallocateTimer(Timer *timer) override {}

  bool StartTimer(Stream *stream, Timer *timer) override;

  bool StopTimer(Stream *stream, Timer *timer) override;

  port::Status BlockHostUntilDone(Stream *stream) override;

  int PlatformDeviceCount() override { return 1; }

  bool DeviceMemoryUsage(int64 *free, int64 *total) const override {
    return false;
  }

  DeviceDescription *PopulateDeviceDescription() const override;

  port::Status EnablePeerAccessTo(StreamExecutorInterface *other) override {
    return port::Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override {
    return true;
  }

  SharedMemoryConfig GetDeviceSharedMemoryConfig() override {
    LOG(INFO) << "Shared memory configuration is unsupported for host "
              << "executors.";
    return SharedMemoryConfig::kDefault;
  }

  port::Status SetDeviceSharedMemoryConfig(SharedMemoryConfig config) override {
    string error_msg{
        "Shared memory configuration is unsupported for host "
        "executors."};
    LOG(INFO) << error_msg;
    return port::Status(port::error::UNIMPLEMENTED, error_msg);
  }

  bool SupportsBlas() const override;
  blas::BlasSupport *CreateBlas() override;

  bool SupportsDnn() const override { return false; }
  dnn::DnnSupport *CreateDnn() override { return nullptr; }

  bool SupportsFft() const override;
  fft::FftSupport *CreateFft() override;

  bool SupportsRng() const override;
  rng::RngSupport *CreateRng() override;

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    LOG(WARNING) << "Events not currently supported by HostExecutor.";
    return nullptr;
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<internal::StreamInterface>(new HostStream());
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(new HostTimer());
  }

  void *GpuContextHack() override { return nullptr; }

 private:
  const PluginConfig plugin_config_;
};

}  // namespace host
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
