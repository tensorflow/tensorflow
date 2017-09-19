/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Declares the InterpreterExecutor class, which is a CPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_EXECUTOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_EXECUTOR_H_

#include <functional>
#include <memory>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/shared_memory_config.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/timer.h"

namespace perftools {
namespace gputools {
namespace interpreter {

using Args = tensorflow::gtl::ArraySlice<DeviceMemoryBase>;

class InterpreterExecutor : public internal::StreamExecutorInterface {
 public:
  explicit InterpreterExecutor(const PluginConfig &plugin_config);
  ~InterpreterExecutor() override;

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

  bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &pop_src,
              uint64 size) override;
  bool Memcpy(Stream *stream, DeviceMemoryBase *pop_dst, const void *host_src,
              uint64 size) override;
  bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *pop_dst,
                            const DeviceMemoryBase &host_src,
                            uint64 size) override {
    return false;
  }

  bool MemZero(Stream *stream, DeviceMemoryBase *location,
               uint64 size) override {
    return false;
  }
  bool Memset(Stream *stream, DeviceMemoryBase *location, uint8 pattern,
              uint64 size) override {
    return false;
  }
  bool Memset32(Stream *stream, DeviceMemoryBase *location, uint32 pattern,
                uint64 size) override {
    return false;
  }

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override { return false; }
  bool SynchronousMemZero(DeviceMemoryBase *location, uint64 size) override {
    return false;
  }

  bool SynchronousMemSet(DeviceMemoryBase *location, int value,
                         uint64 size) override {
    return false;
  }

  port::Status SynchronousMemcpy(DeviceMemoryBase *pop_dst,
                                 const void *host_src, uint64 size) override;
  port::Status SynchronousMemcpy(void *host_dst,
                                 const DeviceMemoryBase &pop_src,
                                 uint64 size) override;
  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *pop_dst,
                                               const DeviceMemoryBase &pop_src,
                                               uint64 size) override {
    return port::Status{port::error::UNIMPLEMENTED, ""};
  }

  bool HostCallback(Stream *stream, std::function<void()> callback) override;

  port::Status AllocateEvent(Event *event) override {
    return port::Status{port::error::UNIMPLEMENTED, ""};
  }

  port::Status DeallocateEvent(Event *event) override {
    return port::Status{port::error::UNIMPLEMENTED, ""};
  }

  port::Status RecordEvent(Stream *stream, Event *event) override {
    return port::Status{port::error::UNIMPLEMENTED, ""};
  }

  port::Status WaitForEvent(Stream *stream, Event *event) override {
    return port::Status{port::error::UNIMPLEMENTED, ""};
  }

  Event::Status PollForEventStatus(Event *event) override {
    return Event::Status::kError;
  }

  bool AllocateStream(Stream *stream) override { return true; }
  void DeallocateStream(Stream *stream) override {}
  bool CreateStreamDependency(Stream *dependent, Stream *other) override;

  bool AllocateTimer(Timer *timer) override { return true; }
  void DeallocateTimer(Timer *timer) override {}
  bool StartTimer(Stream *stream, Timer *timer) override;
  bool StopTimer(Stream *stream, Timer *timer) override;

  bool BlockHostUntilDone(Stream *stream) override;

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
    return SharedMemoryConfig::kDefault;
  }

  port::Status SetDeviceSharedMemoryConfig(SharedMemoryConfig config) override {
    return port::Status{port::error::UNIMPLEMENTED,
                        "Shared memory not supported"};
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<internal::StreamInterface>(new host::HostStream());
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(new host::HostTimer());
  }

 private:
  DeviceMemoryBase AllocateSingleOutput(const xla::Shape &shape);

  port::StatusOr<DeviceMemoryBase> AllocateOutputBuffer(
      const xla::Shape &shape);

  const PluginConfig plugin_config_;
};

}  // namespace interpreter
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_EXECUTOR_H_
