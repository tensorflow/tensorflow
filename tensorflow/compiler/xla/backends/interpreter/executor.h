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

// Declares the XlaInterpreterExecutor class, which is a CPU-only implementation
// of the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_COMPILER_XLA_BACKENDS_INTERPRETER_EXECUTOR_H_
#define TENSORFLOW_COMPILER_XLA_BACKENDS_INTERPRETER_EXECUTOR_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/device_options.h"
#include "tensorflow/compiler/xla/stream_executor/event.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_stream.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_timer.h"
#include "tensorflow/compiler/xla/stream_executor/kernel.h"
#include "tensorflow/compiler/xla/stream_executor/kernel_spec.h"
#include "tensorflow/compiler/xla/stream_executor/launch_dim.h"
#include "tensorflow/compiler/xla/stream_executor/plugin.h"
#include "tensorflow/compiler/xla/stream_executor/rng.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/xla/stream_executor/timer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace stream_executor {
namespace interpreter {

using Args = absl::Span<const DeviceMemoryBase>;

class XlaInterpreterExecutor : public internal::StreamExecutorInterface {
 public:
  explicit XlaInterpreterExecutor(const PluginConfig &plugin_config);
  ~XlaInterpreterExecutor() override;

  tsl::Status Init(int device_ordinal, DeviceOptions device_options) override {
    return ::tsl::OkStatus();
  }

  tsl::Status GetKernel(const MultiKernelLoaderSpec &spec,
                        KernelBase *kernel) override {
    return tsl::errors::Unimplemented("Not Implemented");
  }
  tsl::Status Launch(Stream *stream, const ThreadDim &thread_dims,
                     const BlockDim &block_dims, const KernelBase &kernel,
                     const KernelArgsArrayBase &args) override {
    return tsl::errors::Unimplemented("Not Implemented");
  }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void *GetSubBuffer(DeviceMemoryBase *parent, uint64_t offset_bytes,
                     uint64_t size_bytes) override;
  void Deallocate(DeviceMemoryBase *mem) override;

  void *HostMemoryAllocate(uint64_t size) override { return new char[size]; }
  void HostMemoryDeallocate(void *mem) override {
    delete[] static_cast<char *>(mem);
  }
  bool HostMemoryRegister(void *mem, uint64_t size) override { return true; }
  bool HostMemoryUnregister(void *mem) override { return true; }

  bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &dev_src,
              uint64_t size) override;
  bool Memcpy(Stream *stream, DeviceMemoryBase *dev_dst, const void *host_src,
              uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *pop_dst,
                            const DeviceMemoryBase &host_src,
                            uint64_t size) override {
    return false;
  }

  tsl::Status MemZero(Stream *stream, DeviceMemoryBase *location,
                      uint64_t size) override {
    return tsl::errors::Internal("Interpreter can not memzero");
  }
  tsl::Status Memset(Stream *stream, DeviceMemoryBase *location,
                     uint8_t pattern, uint64_t size) override {
    return tsl::errors::Internal("Interpreter can not memset");
  }
  tsl::Status Memset32(Stream *stream, DeviceMemoryBase *location,
                       uint32_t pattern, uint64_t size) override {
    return tsl::errors::Internal("Interpreter can not memset");
  }

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override { return true; }
  tsl::Status SynchronousMemZero(DeviceMemoryBase *location,
                                 uint64_t size) override {
    return tsl::errors::Internal("Interpreter can not memzero");
  }

  tsl::Status SynchronousMemSet(DeviceMemoryBase *location, int value,
                                uint64_t size) override {
    return tsl::errors::Internal("Interpreter can not memset");
  }

  tsl::Status SynchronousMemcpy(DeviceMemoryBase *dev_dst, const void *host_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpy(void *host_dst, const DeviceMemoryBase &dev_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *pop_dst,
                                              const DeviceMemoryBase &pop_src,
                                              uint64_t size) override {
    return tsl::Status{absl::StatusCode::kUnimplemented, ""};
  }

  bool HostCallback(Stream *stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;

  tsl::Status AllocateEvent(Event *event) override { return ::tsl::OkStatus(); }

  tsl::Status DeallocateEvent(Event *event) override {
    return ::tsl::OkStatus();
  }

  tsl::Status RecordEvent(Stream *stream, Event *event) override {
    return tsl::Status{absl::StatusCode::kUnimplemented, "RecordEvent"};
  }

  tsl::Status WaitForEvent(Stream *stream, Event *event) override {
    return tsl::Status{absl::StatusCode::kUnimplemented, "WaitForEvent"};
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

  tsl::Status BlockHostUntilDone(Stream *stream) override;

  int PlatformDeviceCount() override { return 1; }

  bool DeviceMemoryUsage(int64_t *free, int64_t *total) const override {
    return false;
  }

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(0);
  }

  static tsl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  tsl::Status EnablePeerAccessTo(StreamExecutorInterface *other) override {
    return ::tsl::OkStatus();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override {
    return true;
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
    return std::unique_ptr<internal::StreamInterface>(
        new host::HostStream(/*thread_stack_size=*/0));
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(new host::HostTimer());
  }

 private:
  DeviceMemoryBase AllocateSingleOutput(const xla::Shape &shape);

  tsl::StatusOr<DeviceMemoryBase> AllocateOutputBuffer(const xla::Shape &shape);

  const PluginConfig plugin_config_;
};

}  // namespace interpreter
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_BACKENDS_INTERPRETER_EXECUTOR_H_
